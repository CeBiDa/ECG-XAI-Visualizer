import re

import numpy as np

from ecgproc.constants import FINAL_LEADS
from ecgproc.files import impute_missing
from ecgproc.leads import calculate_missing_leads

_PREFIX = re.compile(r"^(?:MDC_ECG_LEAD|LEAD|ECG)(?:-(?=[A-Za-z0-9])|(?:[\s_:]|-(?![A-Za-z0-9]))+)?",
                     re.IGNORECASE)
_INVERTED = re.compile(r"\binvert|\bnegat", re.IGNORECASE)
_ALIASES = {"DI": "I", "DII": "II", "DIII": "III"}
_FINAL = set(FINAL_LEADS)


def canonical_lead_name(name):
    s = str(name).strip()
    if _INVERTED.search(s):
        return s.upper()
    s = _PREFIX.sub("", s)
    s = re.split(r"[,(\[]", s)[0]
    s = re.sub(r"[\s_]", "", s).upper()
    return _ALIASES.get(s, s)


def standardize_leads(signals, signal_names, derive=True):
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected (n_leads, n_samples) signals, got shape {arr.shape}.")
    names = list(signal_names or [])
    if len(names) != arr.shape[0]:
        raise ValueError(f"{arr.shape[0]} channels but {len(names)} lead names.")

    rows, ignored, duplicates, empty = {}, [], [], []
    for i, raw in enumerate(names):
        key = canonical_lead_name(raw)
        if key not in _FINAL:
            ignored.append(str(raw))
            continue
        if is_empty_channel(arr[i]):
            empty.append(str(raw))
            continue
        if key in rows:
            duplicates.append(str(raw))
            continue
        rows[key] = impute_missing(arr[i])

    original = [ld for ld in FINAL_LEADS if ld in rows]
    calculated = []
    if derive and rows:
        present = list(rows)
        stack = np.vstack([rows[ld] for ld in present])
        stack, new_names, calculated = calculate_missing_leads(stack, present.copy())
        for j, ld in enumerate(new_names):
            rows.setdefault(ld, stack[j])

    n = arr.shape[1]
    out = np.zeros((len(FINAL_LEADS), n), dtype=np.float64)
    missing = []
    for k, ld in enumerate(FINAL_LEADS):
        if ld in rows:
            out[k] = rows[ld]
        else:
            missing.append(ld)

    info = {
        "original_leads": original,
        "calculated_leads": [ld for ld in FINAL_LEADS if ld in calculated],
        "missing_leads": missing,
        "ignored_leads": ignored,
        "duplicate_leads": duplicates,
        "empty_leads": empty,
    }
    return out, info


def is_empty_channel(values):
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    return finite.size == 0 or float(np.ptp(finite)) == 0.0
