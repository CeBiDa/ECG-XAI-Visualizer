"""
Attribution maps: which parts of the input drove the prediction.

Four methods are implemented, three gradient-based and one perturbation-based:

    gradcam     Grad-CAM on the last convolutional stage (layer3): one map over
                time, shared by all leads.
    saliency    |d score / d input| per lead and sample.
    smoothgrad  SmoothGrad x SIGN: the gradient averaged over noisy copies of
                the input, times the sign of the input.
    lime        LIME over 50 ms windows of each lead: copies
                of the record with a few windows set to the lead's median are scored
                by the model (forward passes only), and a weighted ridge regression of
                the log-odds on which windows were kept gives one signed weight per
                window.
"""
import numpy as np

from ecgproc.constants import FINAL_LEADS
from ecgproc.standardize import canonical_lead_name
from ecgxai import segments
from ecgxai.inference import encode, zscore

XAI_METHODS = {
    "gradcam": "Grad-CAM",
    "saliency": "Saliency map",
    "smoothgrad": "SmoothGrad×SIGN",
    "lime": "LIME",
}
_ALIASES = {
    "grad-cam": "gradcam", "grad_cam": "gradcam", "cam": "gradcam",
    "saliency map": "saliency", "saliency maps": "saliency", "saliency_map": "saliency",
    "smoothgrad×sign": "smoothgrad", "smoothgradxsign": "smoothgrad", "smoothgrad_sign": "smoothgrad",
    "smoothgrad-sign": "smoothgrad", "smoothgrad x sign": "smoothgrad", "smoothgradsign": "smoothgrad",
}
XAI_DEFAULTS = {
    "smoothgrad_samples": 50,
    "smoothgrad_noise": 0.2,
    "lime_samples": 1000,
    "seed": 0,
    "top_percentile": 90.0,
}
_SMOOTHGRAD_BATCH = 10
_LIME_BATCH = 10
_LIME_WINDOW_S = 0.05
_LIME_P_OFF = 0.01
_LIME_KERNEL_WIDTH = 0.25
_LIME_ALPHAS = np.geomspace(1e-3, 1e5, 49)
_LIME_FLAT = 1e-6


def method_key(name):
    """'Grad-CAM' / 'gradcam' / 'SmoothGrad×SIGN' / 'LIME' ... -> 'gradcam' | 'saliency' |
    'smoothgrad' | 'lime'."""
    text = str(name or "").strip().lower()
    key = _ALIASES.get(text, text)
    if key not in XAI_METHODS:
        raise ValueError(f"Unknown XAI method '{name}'. Choose from {', '.join(XAI_METHODS)}.")
    return key


def lead_key(name):
    """'II' / 'aVR' / 'Lead V5' -> FINAL_LEADS key; ValueError if not a standard lead."""
    key = canonical_lead_name(name)
    if key not in FINAL_LEADS:
        raise ValueError(f"Unknown lead '{name}'. Choose from I, II, III, aVR, aVL, aVF, V1-V6.")
    return key


def _minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def _score(module, ecg, age_t, sex_t, target):
    out = module(ecg=ecg, age=age_t, gender=sex_t)
    return out[:, target].sum()


def grad_cam(loaded, ecg, age_t, sex_t, target):
    """Grad-CAM over time, (L,) in [0, 1]."""
    import torch
    import torch.nn.functional as F

    module = loaded.module
    layer = getattr(module, loaded.spec.gradcam_layer)
    store = {}

    def hook(_module, _inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = layer.register_forward_hook(hook)
    try:
        module.zero_grad(set_to_none=True)
        with torch.enable_grad():
            _score(module, ecg, age_t, sex_t, target).backward()
    finally:
        handle.remove()                       # hooks must not pile up across calls
        module.zero_grad(set_to_none=True)
    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=2, keepdim=True)
    cam = F.relu((weights * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=ecg.shape[2], mode="linear", align_corners=False)
    return _minmax(cam).squeeze().cpu().numpy()


def saliency(loaded, ecg, age_t, sex_t, target):
    """|gradient| per lead and sample, (12, L) in [0, 1]."""
    import torch

    module = loaded.module
    x = ecg.clone().detach().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    with torch.enable_grad():
        _score(module, x, age_t, sex_t, target).backward()
    module.zero_grad(set_to_none=True)
    sal = x.grad.detach().abs().cpu().squeeze(0).numpy()
    return _minmax(sal)


def smoothgrad_sign(loaded, ecg, age_t, sex_t, target, noise=0.2, samples=50, seed=0,
                    with_zero=False):
    """SmoothGrad x SIGN, (12, L) in [0, 1]; with_zero: (map, where 0 lands in it)."""
    import torch

    module = loaded.module
    samples = max(1, int(samples))
    gen = torch.Generator(device="cpu")
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(int(seed))
    draws = [torch.randn(ecg.shape[1:], generator=gen, dtype=ecg.dtype) for _ in range(samples)]

    total = torch.zeros_like(ecg)
    for i in range(0, samples, _SMOOTHGRAD_BATCH):
        chunk = torch.stack(draws[i:i + _SMOOTHGRAD_BATCH]).to(ecg.device)
        k = chunk.shape[0]
        x = (ecg.expand(k, -1, -1) + chunk * float(noise)).detach().requires_grad_(True)
        module.zero_grad(set_to_none=True)
        with torch.enable_grad():
            _score(module, x, age_t.repeat(k, 1), sex_t.repeat(k, 1), target).backward()
        total += x.grad.detach().sum(dim=0, keepdim=True)
    module.zero_grad(set_to_none=True)
    avg = total / samples
    cam = (avg * torch.sign(ecg)).detach().cpu().squeeze(0).numpy()
    if with_zero:   # the signed map's 0 after the min-max scaling (about 0.5)
        return _minmax(cam), float(-cam.min() / (cam.max() - cam.min() + 1e-8))
    return _minmax(cam)


def _lime_edges(length, fs):
    w = max(1, int(round(_LIME_WINDOW_S * float(fs))))
    n_win = -(-int(length) // w)
    return np.minimum(np.arange(n_win + 1) * w, int(length))


def _ridge_gcv(Z, y, weights):
    n = Z.shape[0]
    total = weights.sum()
    zbar = weights @ Z / total
    ybar = weights @ y / total
    r = np.sqrt(weights * (n / total))
    A = (Z - zbar) * r[:, None]
    b = (y - ybar) * r
    primal = A.shape[1] <= n
    ev, V = np.linalg.eigh(A.T @ A if primal else A @ A.T)
    ev = np.clip(ev, 0.0, None)
    keep = ev > ev.max() * 1e-12
    if primal:
        q = V.T @ (A.T @ b)
        ub2 = np.where(keep, q ** 2 / np.where(keep, ev, 1.0), 0.0)
    else:
        q = V.T @ b
        ub2 = np.where(keep, q ** 2, 0.0)
    outside = max(0.0, float(b @ b - ub2.sum()))
    ev_kept = np.where(keep, ev, 0.0)
    gcv = [n * (outside + float(np.sum((a / (ev + a)) ** 2 * ub2)))
           / max(1e-12, n - 1 - float(np.sum(ev_kept / (ev_kept + a)))) ** 2
           for a in _LIME_ALPHAS]
    coef = V @ (q / (ev + _LIME_ALPHAS[int(np.argmin(gcv))]))
    return coef if primal else A.T @ coef


def _signed_unit(smap, with_zero):
    lo, hi = min(float(smap.min()), 0.0), max(float(smap.max()), 0.0)
    if hi <= lo:
        amap, zero = np.zeros(smap.shape, dtype=np.float32), 0.0
    else:
        amap = ((smap - lo) / (hi - lo)).astype(np.float32)
        zero = float(np.float32(-lo / (hi - lo))) + 0.0
    return (amap, zero) if with_zero else amap


def lime_map(loaded, ecg, age_t, sex_t, target, samples=1000, seed=0, with_zero=False):
    import torch

    x32 = ecg.detach().cpu().squeeze(0).numpy()
    x = x32.astype(np.float64)
    n_leads, length = x.shape
    edges = _lime_edges(length, loaded.spec.fs)
    widths = np.diff(edges)
    n_win = widths.size
    off = np.median(x, axis=1, keepdims=True)
    active = np.logical_or.reduceat(x != off, edges[:-1], axis=1).ravel()
    if not active.any():
        return _signed_unit(np.zeros_like(x), with_zero)

    n = max(2, int(samples))
    gen = torch.Generator(device="cpu")
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(int(seed))
    draws = torch.rand((n - 1, active.size), generator=gen, dtype=torch.float64).numpy()
    kept = np.vstack([np.ones((1, active.size), dtype=bool), draws >= _LIME_P_OFF])  # row 0: as is

    t = int(target)
    y = np.empty(n)
    for i in range(0, n, _LIME_BATCH):
        win = (kept[i:i + _LIME_BATCH] | ~active).reshape(-1, n_leads, n_win)
        copies = np.where(np.repeat(win, widths, axis=2), x, off)     # (k, 12, L)
        renorm = zscore(copies.reshape(-1, length)).reshape(copies.shape)
        renorm[np.ptp(copies, axis=2) == 0] = 0.0
        changed = ~win.all(axis=2)
        batch = np.where(changed[:, :, None], renorm.astype(np.float32), x32)
        k = batch.shape[0]
        xb = torch.from_numpy(batch).to(ecg.device)
        with loaded.lock, torch.no_grad():
            logits = loaded.module(ecg=xb, age=age_t.repeat(k, 1), gender=sex_t.repeat(k, 1))
        logits = logits.detach().cpu().double().numpy()
        y[i:i + k] = logits[:, t] - logits[:, 1 - t]

    if np.ptp(y) <= _LIME_FLAT * max(1.0, float(np.abs(y).max())):
        return _signed_unit(np.zeros_like(x), with_zero)
    Z = kept[:, active].astype(np.float64)
    dist = 1.0 - np.sqrt(Z.sum(axis=1) / Z.shape[1])
    varied = active & ~kept.all(axis=0)
    beta = np.zeros(active.size)
    if varied.any():
        beta[varied] = _ridge_gcv(kept[:, varied].astype(np.float64), y,
                                  np.sqrt(np.exp(-dist ** 2 / _LIME_KERNEL_WIDTH ** 2)))
    return _signed_unit(np.repeat(beta.reshape(n_leads, n_win), widths, axis=1), with_zero)


def attribution_map(loaded, window, age, sex, method, target, options=None, with_zero=False):
    """
    (12, L) float32 attribution in [0, 1] for one model input. with_zero:
    (map, zero), where zero is the level of no attribution (0 except for the
    signed SmoothGrad x SIGN and LIME maps).
    """
    opts = dict(XAI_DEFAULTS)
    opts.update({k: v for k, v in (options or {}).items() if k in XAI_DEFAULTS})
    key = method_key(method)
    ecg, age_t, sex_t = encode(loaded, window, age, sex)
    zero = 0.0
    if key == "lime":
        amap, zero = lime_map(loaded, ecg, age_t, sex_t, target,
                              samples=int(opts["lime_samples"]), seed=opts["seed"],
                              with_zero=True)
    else:
        with loaded.lock:
            if key == "gradcam":
                cam = grad_cam(loaded, ecg, age_t, sex_t, target)
                amap = np.tile(np.asarray(cam, dtype=np.float64)[None, :], (ecg.shape[1], 1))
            elif key == "saliency":
                amap = saliency(loaded, ecg, age_t, sex_t, target)
            else:
                amap, zero = smoothgrad_sign(loaded, ecg, age_t, sex_t, target,
                                             noise=float(opts["smoothgrad_noise"]),
                                             samples=int(opts["smoothgrad_samples"]),
                                             seed=opts["seed"], with_zero=True)
    amap = np.asarray(amap, dtype=np.float32)
    return (amap, zero) if with_zero else amap


def explain(loaded, window, age, sex, target, method, lead, model_record=None,
            start=0, options=None, attribution=None, zero=0.0, delineations=None):
    opts = dict(XAI_DEFAULTS)
    opts.update({k: v for k, v in (options or {}).items() if k in XAI_DEFAULTS})
    key, lead = method_key(method), lead_key(lead)
    if attribution is None:
        attribution, zero = attribution_map(loaded, window, age, sex, key, target, opts,
                                            with_zero=True)
    idx = FINAL_LEADS.index(lead)
    if model_record is None:
        model_record, start = window, 0
    signal = np.asarray(model_record)[idx]
    if delineations is not None and lead not in delineations:
        delineations[lead] = segments.delineate_safe(signal, loaded.spec.fs)
    seg, labels = segments.analyze(signal, loaded.spec.fs, attribution[idx], start,
                                   float(opts["top_percentile"]), zero,
                                   None if delineations is None else delineations[lead])
    return {
        "method": key,
        "method_name": XAI_METHODS[key],
        "lead": lead,
        "target": int(target),
        "target_label": loaded.spec.label(target),
        "map": attribution,
        "segments": seg,
        "labels": labels,
    }