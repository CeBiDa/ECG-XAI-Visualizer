import re
import numpy as np
from pathlib import Path

def find_files(input_path, extensions):
    """
    Finds all relevant files in the given path with specified extensions.
    """
    input_path = Path(input_path)
    wanted = {ext.lower() for ext in extensions}
    if input_path.is_file():
        if input_path.suffix.lower() not in wanted:
            return []
        if input_path.suffix.lower() == ".mat" and ".hea" in wanted:
            for header in (input_path.with_suffix(".hea"), input_path.with_suffix(".HEA")):
                if header.is_file():
                    return [header]
        return [input_path]

    all_files = sorted(f for f in input_path.rglob("*")
                       if f.is_file() and f.suffix.lower() in wanted
                       and not f.name.startswith("._") and "__MACOSX" not in f.parts)

    final_files_to_process = []
    processed_wfdb_stems = set()
    wfdb_extensions = {".dat", ".hea"}
    headers = {(f.parent, f.stem) for f in all_files if f.suffix.lower() == ".hea"}

    for file in all_files:
        if file.suffix.lower() == ".mat" and (file.parent, file.stem) in headers:
            continue
        if file.suffix.lower() in wfdb_extensions:
            key = (file.parent, file.stem)
            if key not in processed_wfdb_stems:
                processed_wfdb_stems.add(key)
                final_files_to_process.append(file)
        else:
            final_files_to_process.append(file)

    return final_files_to_process

def zero_pad(signal, target_len):
    if len(signal) > target_len:
        return signal[:target_len]
    padded = np.zeros(target_len)
    padded[:len(signal)] = signal
    return padded

def impute_missing(signal):
    sig = np.asarray(signal, dtype=float).copy()
    bad = ~np.isfinite(sig)
    if bad.any():
        good = ~bad
        if not good.any():
            return np.zeros_like(sig)
        sig[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), sig[good])
        sig[~np.isfinite(sig)] = 0
    return sig

def normalize_time(value: str) -> str:
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 6:
        hh = digits[0:2]
        mm = digits[2:4]
        ss = digits[4:6]
        frac = digits[6:] if len(digits) > 6 else ""
        if frac:
            return f"{hh}{mm}{ss}.{frac}"
        else:
            return f"{hh}{mm}{ss}"
    return digits
