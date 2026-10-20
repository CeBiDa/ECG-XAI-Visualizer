import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ecgproc.constants import FINAL_LEADS, SUPPORTED_EXTENSIONS, lead_label
from ecgproc.files import find_files
from ecgproc.patient import load_metadata_csv, resolve
from ecgproc.readers import read_asc, read_csv, read_dat_hea, read_dcm, read_mat, read_xml
from ecgproc.standardize import canonical_lead_name, is_empty_channel
from runner.execution_runner import is_summary_file

READERS = {
    ".mat": read_mat, ".dat": read_dat_hea, ".hea": read_dat_hea,
    ".csv": read_csv, ".asc": read_asc, ".dcm": read_dcm, ".xml": read_xml,
    ".hl7": read_xml,   # read_xml detects the AnnotatedECG schema itself
}

META_KEYS = ["source", "patient_id", "patient_sex", "patient_age", "patient_birthdate",
             "exam_date", "exam_time", "patient_weight", "patient_height", "pacemaker"]


def record_id(path, target):
    root = target if target.is_dir() else target.parent
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    if path.suffix.lower() in (".dat", ".hea"):
        rel = rel.with_suffix(".hea")
    return rel.as_posix()


def inspect(path, table=None, default_fs=None, rid=None):
    rid = rid or path.name
    print(f"\n=== {rid} ({path.stat().st_size / 1024:.1f} KB) ===")
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        print(f"  no reader for '{path.suffix}'")
        return False
    try:
        signals, meta = reader(path)
    except Exception as e:
        print(f"  READ FAILED: {e.__class__.__name__}: {e}")
        return False
    names = [str(s) for s in meta.get("signal_names", [])]
    print(f"  shape: {signals.shape[0]} leads x {signals.shape[1]} samples")
    fs = meta.get("fs")
    print(f"  fs: {fs}" + ("" if fs else (f"  (none stored; --fs {default_fs} would be used)"
                                          if default_fs else "  (none stored: needs --fs)")))
    import pandas as pd
    rows = [pd.to_numeric(pd.Series(r), errors="coerce").to_numpy(dtype=float) for r in signals]
    mapped, empty = {}, set()
    for i, n in enumerate(names):
        key = canonical_lead_name(n)
        mapped[n] = key if key in FINAL_LEADS else None
        if mapped[n] and i < len(rows) and is_empty_channel(rows[i]):
            mapped[n] = None
            empty.add(n)
    shown = ", ".join(f"{n} (flat/empty: not recorded)" if n in empty else
                      (f"{n}->{lead_label(k)}" if k and k != n.upper() else (n if k else f"{n} (ignored)"))
                      for n, k in mapped.items())
    print(f"  leads: {shown}")
    present = {k for k in mapped.values() if k}
    absent = [lead_label(ld) for ld in FINAL_LEADS if ld not in present]
    if absent:
        print(f"  not recorded: {', '.join(absent)} (limb leads are derived when two of I/II/III exist)")
    filled = {k: meta[k] for k in META_KEYS if meta.get(k) not in (None, "")}
    print("  metadata: " + (", ".join(f"{k}={v}" for k, v in filled.items()) if filled else "none found"))
    who = resolve(meta, path, table, record_id=rid)
    age = f"{who['age']:g} ({who['age_source']}: {who['age_detail']})" if who["age"] is not None else "unknown"
    sex = f"{who['sex']} ({who['sex_source']}: {who['sex_detail']})" if who["sex"] is not None else "unknown"
    print(f"  model inputs: age {age}, sex {sex}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Inspect ECG files without analysing them.")
    parser.add_argument("path", nargs="?", default=str(project_root / "data" / "input_ecgs"))
    parser.add_argument("--metadata", help="ID / age / sex CSV, to check the matching")
    parser.add_argument("--fs", type=float, help="Fallback sampling rate you plan to pass")
    args = parser.parse_args()

    target = Path(args.path)
    # Not inputs, as in the analysis: the metadata CSV and earlier summaries.
    meta_csv = Path(args.metadata).resolve() if args.metadata else None
    files = [f for f in find_files(target, SUPPORTED_EXTENSIONS)
             if not is_summary_file(f.name) and f.resolve() != meta_csv]
    if not files:
        print(f"No supported ECG files under {target}")
        return 1
    table = load_metadata_csv(args.metadata) if args.metadata else None
    ok = sum(inspect(f, table, args.fs, record_id(f, target)) for f in files)
    print(f"\n{ok}/{len(files)} files readable.")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
