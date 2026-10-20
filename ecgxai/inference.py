import numpy as np

from ecgproc.preprocess import resample_signal


def to_model_rate(signals, fs, spec, method="fft"):
    x = np.asarray(signals, dtype=np.float64)
    fs = float(fs or 0)
    if fs <= 0:
        raise ValueError("A sampling rate is needed to build the model input.")
    if abs(fs - spec.fs) < 1e-6:
        return x, False
    return np.asarray(resample_signal(x, fs, spec.fs, method=method), dtype=np.float64), True


def window_bounds(n_samples, spec, window_start=0.0):
    want = spec.window_samples
    start = int(round(max(0.0, float(window_start or 0.0)) * spec.fs))
    notes = []
    if n_samples <= want:
        if start > 0:
            notes.append(f"window start {float(window_start):g} s ignored, record is "
                         f"{n_samples / spec.fs:.3f} s")
        if n_samples < want:
            notes.append(f"record is {n_samples / spec.fs:.3f} s, shorter than the "
                         f"{spec.window_s:g} s window the model was trained on")
        return 0, n_samples, notes
    if start + want > n_samples:
        new_start = n_samples - want
        notes.append(f"window moved to start at {new_start / spec.fs:.3f} s "
                     f"(record ends at {n_samples / spec.fs:.3f} s)")
        start = new_start
    return start, start + want, notes


def zscore(window):
    w = np.asarray(window, dtype=np.float64)
    mean = w.mean(axis=1, keepdims=True)
    std = w.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (w - mean) / std


def encode(loaded, window, age, sex):
    import torch

    spec = loaded.spec
    ecg = torch.from_numpy(zscore(window)).float().unsqueeze(0).to(loaded.device)
    age_t = torch.tensor([[(float(age) - spec.age_mean) / spec.age_std]],
                         dtype=torch.float32, device=loaded.device)
    sex_t = torch.tensor([[spec.sex_code(sex)]], dtype=torch.float32, device=loaded.device)
    return ecg, age_t, sex_t


def predict(loaded, window, age, sex, threshold=None):
    import torch

    spec = loaded.spec
    thr = spec.threshold if threshold is None else float(threshold)
    ecg, age_t, sex_t = encode(loaded, window, age, sex)
    with loaded.lock, torch.no_grad():
        logits = loaded.module(ecg=ecg, age=age_t, gender=sex_t)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    p = float(probs[spec.positive_class])
    cls = spec.positive_class if p >= thr else 1 - spec.positive_class
    return {
        "probability": p,
        "threshold": thr,
        "class": int(cls),
        "label": spec.label(cls),
        "probabilities": [float(v) for v in probs],
    }
