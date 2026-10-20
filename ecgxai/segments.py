import warnings

import numpy as np

SEGMENTS = ("P_Wave", "QRS_Complex", "ST_Segment", "T_Wave")
_INDEX = {name: i for i, name in enumerate(SEGMENTS)}

FALLBACK_WINDOWS = {
    "P_Wave": (-0.20, -0.06),
    "QRS_Complex": (-0.05, 0.05),
    "ST_Segment": (0.05, 0.15),
    "T_Wave": (0.15, 0.40),
}
_PAINT_ORDER = ("P_Wave", "T_Wave", "ST_Segment", "QRS_Complex")
_METHODS = ("dwt", "prominence")
_MIN_BOUNDED = 0.5

# Boundary keys
_BOUNDS = {
    "P_Wave": ("ECG_P_Onsets", "ECG_P_Offsets"),
    "QRS_Complex": ("ECG_R_Onsets", "ECG_R_Offsets"),
    "ST_Segment": ("ECG_R_Offsets", "ECG_T_Onsets"),
    "T_Wave": ("ECG_T_Onsets", "ECG_T_Offsets"),
}


def _value(waves, key, i):
    vals = waves.get(key) if waves else None
    if vals is None or i >= len(vals):
        return None
    v = vals[i]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else int(v)


def _segment_masks(waves, n_beats, n):
    masks = {name: np.zeros(n, dtype=bool) for name in SEGMENTS}
    bounded = {name: 0 for name in SEGMENTS}
    for i in range(n_beats):
        for name, (on_key, off_key) in _BOUNDS.items():
            on, off = _value(waves, on_key, i), _value(waves, off_key, i)
            if on is not None and off is not None and 0 <= on < off <= n:
                masks[name][on:off] = True
                bounded[name] += 1
    return masks, bounded


def delineate(lead, fs):
    import neurokit2 as nk

    sig = np.asarray(lead, dtype=np.float64)
    n = sig.size
    fs = float(fs)
    if n == 0 or not np.isfinite(sig).all() or np.ptp(sig) <= 0:
        raise ValueError("lead is flat or empty")

    prepared = {}

    def prepare(polarity):
        if polarity not in prepared:
            cleaned = nk.ecg_clean(polarity * sig, sampling_rate=fs)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
            rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            prepared[polarity] = (cleaned, info, rpeaks)
        return prepared[polarity]

    chosen, rejected = None, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 'R peaks' on the S waves.
        cleaned, _, rpeaks = prepare(1)
        half = max(1, int(0.06 * fs))
        extremes = [s[np.argmax(np.abs(s))] for s in
                    (cleaned[max(0, r - half):r + half + 1] for r in rpeaks) if s.size]
        polarities = (1, -1) if not extremes or np.median(extremes) < 0 else (1,)
        for candidate in _METHODS:
            for polarity in polarities:
                cleaned, info, rpeaks = prepare(polarity)
                if rpeaks.size == 0:
                    continue
                try:
                    _, waves = nk.ecg_delineate(cleaned, info, sampling_rate=fs, method=candidate)
                except Exception:
                    continue
                masks, bounded = _segment_masks(waves, rpeaks.size, n)
                if min(bounded["QRS_Complex"], bounded["T_Wave"]) >= _MIN_BOUNDED * rpeaks.size:
                    chosen = (candidate, rpeaks, masks, bounded)
                    break
                if candidate not in rejected:
                    rejected.append(candidate)
            if chosen is not None:
                break

    note = ""
    if chosen is not None:
        method, rpeaks, masks, bounded = chosen
        if bounded["ST_Segment"] < _MIN_BOUNDED * rpeaks.size:
            note = (f"ST segment bounded in only {bounded['ST_Segment']} of {rpeaks.size} beats; "
                    "its share is a lower bound")
    else:
        rpeaks = prepared[1][2] if prepared[1][2].size else prepare(-1)[2]
        if rpeaks.size == 0:
            raise ValueError("no R peaks detected")
        method = "window"
        if rejected:
            note = f"{', '.join(rejected)} left QRS or T unbounded; fixed windows used"
        elif n < 4 * fs:
            note = "record shorter than the 4 s NeuroKit2 needs; fixed windows used"
        else:
            note = "delineation failed; fixed windows used"
        masks = {name: np.zeros(n, dtype=bool) for name in SEGMENTS}
        for r in rpeaks:
            for name, (lo, hi) in FALLBACK_WINDOWS.items():
                a = max(0, int(r + lo * fs))
                b = min(n, int(r + hi * fs))
                if b > a:
                    masks[name][a:b] = True

    labels = np.full(n, -1, dtype=np.int8)
    for name in _PAINT_ORDER:
        labels[masks[name]] = _INDEX[name]
    return labels, method, int(rpeaks.size), note


def delineate_safe(lead, fs):
    try:
        return delineate(lead, fs)
    except Exception as e:
        return str(e) or type(e).__name__


def attribution_shares(attribution, labels, top_percentile=90.0, zero=0.0):
    attr = np.asarray(attribution, dtype=np.float64)
    labels = np.asarray(labels)
    if attr.shape != labels.shape:
        raise ValueError(f"attribution {attr.shape} and labels {labels.shape} differ in length")
    thresh = np.percentile(attr, top_percentile)
    top = np.where((attr >= thresh) & (attr > zero), attr - zero, 0.0)
    total = top.sum()
    if total <= 0:
        return None, f"no positive attribution in the top {100 - top_percentile:g} % of this lead"
    shares = {name: round(float(top[labels == i].sum() / total * 100.0), 1)
              for i, name in enumerate(SEGMENTS)}
    shares["Other"] = round(float(top[labels < 0].sum() / total * 100.0), 1)
    return shares, ""


def analyze(lead_record, fs, attribution, start, top_percentile=90.0, zero=0.0, delineation=None):
    attribution = np.asarray(attribution, dtype=np.float64)
    length = attribution.size
    result = {"shares": None, "method": "", "n_beats": 0,
              "top_percentile": float(top_percentile), "zero": float(zero), "note": "",
              "lead_note": "", "share_note": ""}
    if delineation is None:
        delineation = delineate_safe(lead_record, fs)
    if isinstance(delineation, str):
        result["note"] = result["lead_note"] = f"segmentation failed: {delineation}"
        return result, None
    labels, method, n_beats, note = delineation
    window_labels = labels[start:start + length]
    if window_labels.size != length:          # window runs past the record end
        window_labels = np.pad(window_labels, (0, length - window_labels.size), constant_values=-1)
    shares, share_note = attribution_shares(attribution, window_labels, top_percentile, zero)
    result.update(shares=shares, method=method, n_beats=n_beats, lead_note=note, share_note=share_note,
                  note="; ".join(x for x in (note, share_note) if x))
    return result, window_labels
