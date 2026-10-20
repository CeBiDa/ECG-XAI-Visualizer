import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ecgproc.constants import FINAL_LEADS, lead_label
from ecgproc.files import impute_missing
from ecgproc.patient import resolve as resolve_patient
from ecgproc.preprocess import EMD_DEFAULTS, preprocess_signals, resample_signal
from ecgproc.readers import read_asc, read_csv, read_dat_hea, read_dcm, read_mat, read_xml
from ecgproc.standardize import canonical_lead_name, is_empty_channel, standardize_leads
from ecgxai.explain import XAI_DEFAULTS, XAI_METHODS, explain, method_key
from ecgxai.inference import predict, to_model_rate, window_bounds
from ecgxai.plotting import probability_text, save_xai_figure, xai_titles

READERS = {
    ".mat": read_mat, ".dat": read_dat_hea, ".hea": read_dat_hea,
    ".csv": read_csv, ".asc": read_asc, ".dcm": read_dcm,
    ".xml": read_xml, ".hl7": read_xml,   # read_xml recognises HL7 aECG itself
}
FORMAT_NAMES = {"mat": "MATLAB", "wfdb": "WFDB", "csv": "CSV", "asc": "ASC",
                "dcm": "DICOM", "xml": "XML", "hl7": "HL7 aECG"}

SHARE_COLUMNS = {
    "P_Wave": "p_wave",
    "QRS_Complex": "qrs",
    "ST_Segment": "st_segment",
    "T_Wave": "t_wave",
    "Other": "other",
}


@dataclass
class AnalysisOptions:
    pipeline: list = field(default_factory=list)
    notch: int = 50
    default_fs: Optional[float] = None
    resample: Optional[float] = None
    resample_method: str = "fft"
    derive_leads: bool = True
    anonymize: bool = False
    emd_options: dict = field(default_factory=lambda: dict(EMD_DEFAULTS))
    threshold: Optional[float] = None
    window_start: float = 0.0
    allow_missing_leads: bool = False
    fallback_age: Optional[float] = None
    fallback_sex: Optional[str] = None
    xai_methods: tuple = ("gradcam",)       # method keys, in order; () = prediction only
    xai_lead: str = "II"
    xai_options: dict = field(default_factory=lambda: dict(XAI_DEFAULTS))
    save_plots: bool = True
    save_maps: bool = False


def _as_float(signals):
    try:
        return np.asarray(signals, dtype=np.float64)
    except (TypeError, ValueError):
        import pandas as pd
        rows = np.asarray(signals, dtype=object)
        if rows.ndim != 2:
            raise
        return np.vstack([pd.to_numeric(pd.Series(r), errors="coerce").to_numpy(dtype=np.float64)
                          for r in rows])


def _positive(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 and np.isfinite(v) else None


def _num(value, digits=4):
    """Compact number for the summary: 500.0 -> 500, 0.123456 -> 0.1235."""
    if value is None or value == "":
        return ""
    v = round(float(value), digits)
    return int(v) if v == int(v) else v


def _labels(keys):
    return ",".join(lead_label(k) for k in keys)


def pipeline_text(opts, steps=None):
    steps = list(opts.pipeline if steps is None else steps)
    if opts.resample:
        steps.append(f"resample {_num(opts.resample)} Hz ({opts.resample_method})")
    return ",".join(steps)


def share_text(shares):
    short = {"P_Wave": "P", "QRS_Complex": "QRS", "ST_Segment": "ST", "T_Wave": "T", "Other": "other"}
    return " · ".join(f"{short[k]} {shares[k]:.1f} %" for k in short)


def analyze_file(file_path, record_id, out_base, opts, loaded, output_dir,
                 patient_table=None, log=print, keep=False):
    file_path = Path(file_path)
    rid = record_id
    spec = loaded.spec

    def result(status, reason="", row=None, record=None, outputs=()):
        return {"status": status, "file": rid, "reason": reason, "row": row,
                "record": record, "outputs": list(outputs)}

    # 1. read
    reader = READERS.get(file_path.suffix.lower())
    if reader is None:
        return result("skipped", f"unsupported extension '{file_path.suffix}'")
    try:
        signals, meta = reader(file_path)
    except Exception as e:
        log(f"Error reading {rid}: {e}")
        return result("failed", f"read error: {e}")
    meta = dict(meta or {})
    raw = _as_float(signals)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] < 2:
        log(f"Error reading {rid}: no usable samples (shape {raw.shape}).")
        return result("failed", "no usable samples")
    names = [str(n) for n in (meta.get("signal_names") or [])]
    if len(names) != raw.shape[0]:
        names = [f"lead{i + 1}" for i in range(raw.shape[0])]

    # 2. sampling rate
    file_fs = _positive(meta.get("fs"))
    fs = file_fs or _positive(opts.default_fs)
    fs_source = "file" if file_fs else ("fallback" if fs else "")
    if not fs:
        log(f"Skipping {rid}: unknown sampling rate (set a fallback rate / --fs).")
        return result("skipped", "unknown sampling rate")

    # 3. age and sex
    patient = resolve_patient(meta, file_path, patient_table, opts.fallback_age, opts.fallback_sex,
                              reveal_ids=not opts.anonymize, record_id=rid)
    lacking = [f for f in ("age", "sex") if patient[f] is None]
    if lacking:
        what = " and ".join(lacking)
        log(f"Skipping {rid}: no {what} in the file, the metadata CSV or the fallback values.")
        return result("skipped", f"no {what}")
    if opts.anonymize:
        for key in ("patient_id", "patient_firstname", "patient_lastname", "patient_birthdate",
                    "exam_date", "exam_time"):
            meta[key] = ""

    # 4. standard channels only
    padded = meta.get("padded_leads") or {}
    real = [int(padded.get(name, raw.shape[1])) for name in names]
    keep_idx, keys, ignored, duplicates, empty = [], [], [], [], []
    for i, name in enumerate(names):
        key = canonical_lead_name(name)
        if key not in FINAL_LEADS:
            ignored.append(name)
        elif real[i] < 2 or is_empty_channel(raw[i][:real[i]]):
            empty.append(name)
        elif key in keys:
            duplicates.append(name)
        else:
            keep_idx.append(i)
            keys.append(key)
    early_notes = []
    if keep_idx and padded:
        need = int(round(spec.window_s * float(fs)))
        longest = max(real[i] for i in keep_idx)
        short = [i for i in keep_idx if real[i] < need] if longest >= need else []
        if short:
            early_notes.append(f"lead(s) {', '.join(names[i] for i in short)} cut short by the "
                               "export (shorter than the model window): treated as not recorded")
            keys = [k for i, k in zip(keep_idx, keys) if i not in short]
            keep_idx = [i for i in keep_idx if i not in short]
    if not keys:
        found = ", ".join(names[:16])
        why = "no standard lead with signal" if (empty or early_notes) else "no standard lead names"
        log(f"Skipping {rid}: {why} among {found}.")
        return result("skipped", why)
    common = min(real[i] for i in keep_idx)
    selected = np.vstack([impute_missing(raw[i][:common]) for i in keep_idx])
    cut = [names[i] for i in keep_idx if real[i] > common]
    if cut:
        early_notes.append(f"lead(s) {', '.join(names[i] for i in keep_idx if real[i] == common)} "
                           f"end early (truncated export): record cut to the "
                           f"{common / float(fs):.2f} s all leads have")
    for note in early_notes:
        log(f"{rid}: {note}.")

    # 5. pre-processing at the file's own rate
    proc, proc_fs = selected, float(fs)
    steps = list(opts.pipeline)
    if "notch" in steps and float(opts.notch) >= proc_fs / 2:
        steps = [s for s in steps if s != "notch"]
        early_notes.append(f"notch {opts.notch} Hz skipped: not below the Nyquist frequency "
                           f"({proc_fs / 2:g} Hz) of this record")
        log(f"{rid}: {early_notes[-1]}.")
    if steps:
        if "emd" in steps:
            log(f"Processing {rid} with EMD (this may take a while)...")
        try:
            proc = preprocess_signals(proc, steps, proc_fs, opts.notch,
                                      emd_options=opts.emd_options, log=log)
        except Exception as e:
            log(f"Error pre-processing {rid}: {e}")
            return result("failed", f"pre-processing error: {e}")
    model_source = None
    if opts.resample and abs(float(opts.resample) - proc_fs) > 1e-6:
        if abs(proc_fs - spec.fs) < 1e-6 and float(opts.resample) > spec.fs:
            model_source = proc
        try:
            proc = resample_signal(proc, proc_fs, float(opts.resample), method=opts.resample_method)
            proc_fs = float(opts.resample)
        except Exception as e:
            log(f"Error resampling {rid}: {e}")
            return result("failed", f"resampling error: {e}")

    # 6. fixed 12-lead layout
    std, leads = standardize_leads(proc, keys, derive=opts.derive_leads)
    leads["ignored_leads"], leads["duplicate_leads"] = ignored, duplicates
    leads["empty_leads"] = empty
    if leads["missing_leads"] and not opts.allow_missing_leads:
        miss = _labels(leads["missing_leads"])
        limb = any(ld in leads["missing_leads"] for ld in FINAL_LEADS[:6])
        why = ("" if not limb else " (deriving limb leads is off)" if not opts.derive_leads
               else "; limb leads are derived only from two of I, II, III")
        log(f"Skipping {rid}: lead(s) {miss} not recorded{why}. The model needs all 12; allow "
            "missing leads to run it on zero-filled ones.")
        return result("skipped", f"missing leads {miss}")

    # 7. model window and prediction
    try:
        if model_source is not None:
            model_rec, resampled = standardize_leads(model_source, keys, derive=opts.derive_leads)[0], True
        else:
            model_rec, resampled = to_model_rate(std, proc_fs, spec, method=opts.resample_method)
        start, stop, notes = window_bounds(model_rec.shape[1], spec, opts.window_start)
        notes = early_notes + notes
        window = model_rec[:, start:stop]
        pred = predict(loaded, window, patient["age"], patient["sex"], opts.threshold)
    except Exception as e:
        log(f"Error running the model on {rid}: {e}")
        log(traceback.format_exc().rstrip())
        return result("failed", f"model error: {e}")
    if leads["missing_leads"]:
        notes.append(f"zero-filled leads {_labels(leads['missing_leads'])}")
    if ignored:
        notes.append(f"ignored channels {', '.join(ignored)}")
    if duplicates:
        notes.append(f"duplicate channels {', '.join(duplicates)}")
    if empty:
        notes.append(f"empty or flat channels treated as not recorded: {', '.join(empty)}")

    # 8. explanation of the reported class
    methods = tuple(dict.fromkeys(method_key(m) for m in opts.xai_methods or ()))
    xai_lead = lead_label(canonical_lead_name(opts.xai_lead))
    results, xai_notes, delineations = {}, {}, {}
    for m in methods:
        try:
            results[m] = explain(loaded, window, patient["age"], patient["sex"], pred["class"],
                                 m, opts.xai_lead, model_record=model_rec, start=start,
                                 options=opts.xai_options, delineations=delineations)
            xai_notes[m] = results[m]["segments"]["note"]
        except Exception as e:
            xai_notes[m] = f"explanation failed: {e}"
            log(f"{rid}: {XAI_METHODS[m]} explanation failed: {e}")

    # 9. files
    outputs, plot_files, maps_files = [], {}, {}
    positive = pred["class"] == spec.positive_class
    output_dir = Path(output_dir)
    for m, xai in results.items():
        stem = f"{out_base}_xai" if len(methods) == 1 else f"{out_base}_xai_{m}"
        idx = FINAL_LEADS.index(xai["lead"])
        if opts.save_plots:
            path = output_dir / f"{stem}.png"
            try:
                title, subtitle, color = xai_titles(rid, lead_label(xai["lead"]), xai["method_name"],
                                                    pred, xai["target_label"], positive)
                save_xai_figure(path, window[idx], spec.fs, xai["map"][idx], xai["labels"],
                                xai["segments"], title, subtitle, color,
                                xai["segments"]["top_percentile"], t0=start / spec.fs)
                plot_files[m] = path.name
                outputs.append(str(path))
            except Exception as e:
                log(f"{rid}: could not write {path.name}: {e}")
        if opts.save_maps:
            path = output_dir / f"{stem}.npz"
            try:
                np.savez_compressed(
                    path, model_input=window.astype(np.float32), attribution=xai["map"],
                    segment_labels=(xai["labels"] if xai["labels"] is not None
                                    else np.full(window.shape[1], -1, dtype=np.int8)),
                    leads=np.array([lead_label(k) for k in FINAL_LEADS]),
                    fs=spec.fs, window_start_s=start / spec.fs,
                    probability=pred["probability"], threshold=pred["threshold"],
                    method=xai["method"], target=xai["target"],
                    zero_level=xai["segments"].get("zero", 0.0), xai_note=xai_notes[m],
                    lead=lead_label(xai["lead"]), top_percentile=xai["segments"]["top_percentile"])
                maps_files[m] = path.name
                outputs.append(str(path))
            except Exception as e:
                log(f"{rid}: could not write {path.name}: {e}")

    # 10. summary row
    segs = {m: x["segments"] for m, x in results.items()}
    seg = next(iter(segs.values()), {})
    texts = {m: share_text(segs[m]["shares"]) if segs.get(m, {}).get("shares") else xai_notes[m]
             for m in methods}
    own = [(m, segs[m]["share_note"] if m in segs else xai_notes[m]) for m in methods]
    xai_note = [seg.get("lead_note", "")] + [f"{XAI_METHODS[m]}: {note}" for m, note in own if note]
    row = {
        "filename": rid,
        "patient_id": meta.get("patient_id", ""),
        "exam_date": meta.get("exam_date", ""),
        "exam_time": meta.get("exam_time", ""),
        "patient_age": _num(patient["age"], 2),
        "age_source": patient["age_source"],
        "patient_sex": patient["sex"],
        "sex_source": patient["sex_source"],
        "sampling_freq": _num(fs),
        "fs_source": fs_source,
        "n_samples": int(selected.shape[1]),
        "original_leads": _labels(leads["original_leads"]),
        "calculated_leads": _labels(leads["calculated_leads"]),
        "missing_leads": _labels(leads["missing_leads"]),
        "pipeline": pipeline_text(opts, steps),
        "model": spec.name,
        "window_start_s": _num(start / spec.fs, 3),
        "probability": _num(pred["probability"], 6),
        "threshold": _num(pred["threshold"], 10),
        "prediction": pred["label"],
        "xai_methods": ",".join(methods),
        "xai_lead": xai_lead if methods else "",
        "xai_target": next(iter(results.values()))["target_label"] if results else "",
        **{f"{m}_{col}": (segs[m]["shares"] or {}).get(name, "") if m in segs else ""
           for m in XAI_METHODS for name, col in SHARE_COLUMNS.items()},
        "delineation": seg.get("method", ""),
        "xai_note": "; ".join(x for x in xai_note if x),
        "notes": "; ".join(notes),
        **{f"{m}_plot": plot_files.get(m, "") for m in XAI_METHODS},
    }

    line = f"{rid}: {pred['label']} (p = {probability_text(pred['probability'], pred['threshold'])})"
    if len(methods) > 1:
        line = "\n".join([f"{line} · lead {xai_lead}"]
                         + [f"    {XAI_METHODS[m]}: {texts[m]}" for m in methods])
    elif results:
        line += f" · {XAI_METHODS[methods[0]]} lead {xai_lead}: {texts[methods[0]]}"
    log(line)

    record = None
    if keep:
        record = {
            "id": rid,
            "path": str(file_path),
            "anonymized": bool(opts.anonymize),
            "format": FORMAT_NAMES.get(meta.get("source", ""), file_path.suffix.lstrip(".").upper()),
            "meta": {k: v for k, v in meta.items() if k not in ("signal_names", "comments")},
            "patient": patient,
            "raw": {"signals": raw.astype(np.float32), "names": names, "fs": float(fs),
                    "fs_source": fs_source},
            "processed": {"signals": std.astype(np.float32), "fs": proc_fs},
            "model_record": model_rec.astype(np.float32) if resampled else None,
            "window": {"start": int(start), "stop": int(stop), "fs": spec.fs},
            "leads": leads,
            "pipeline": pipeline_text(opts, steps),
            "prediction": pred,
            "model": spec.key,
            "xai": {(m, x["lead"]): x for m, x in results.items()},
            "xai_note": "; ".join((f"{XAI_METHODS[m]}: " if len(methods) > 1 else "") + xai_notes[m]
                                  for m in methods if m not in results),
            "delineations": delineations,
            "notes": notes,
            "outputs": {"plot": plot_files, "maps": maps_files},
            "options": {"xai_options": dict(opts.xai_options)},
        }
    return result("success", row=row, record=record, outputs=outputs)


def model_record_of(record):
    rec = record.get("model_record")
    return rec if rec is not None else record["processed"]["signals"]


def model_window_of(record):
    w = record["window"]
    return np.asarray(model_record_of(record)[:, w["start"]:w["stop"]], dtype=np.float64)


def explain_record(loaded, record, method, lead, options=None):
    from ecgxai.explain import lead_key

    key, lead = method_key(method), lead_key(lead)
    cache = record.setdefault("xai", {})
    if (key, lead) in cache:
        return cache[(key, lead)]
    reuse, zero = next(((v["map"], v["segments"].get("zero", 0.0))
                        for (m, _), v in list(cache.items()) if m == key), (None, 0.0))
    opts = dict(record.get("options", {}).get("xai_options") or XAI_DEFAULTS)
    opts.update(options or {})
    window = model_window_of(record)
    res = explain(loaded, window, record["patient"]["age"], record["patient"]["sex"],
                  record["prediction"]["class"], key, lead,
                  model_record=np.asarray(model_record_of(record), dtype=np.float64),
                  start=record["window"]["start"], options=opts, attribution=reuse, zero=zero,
                  delineations=record.setdefault("delineations", {}))
    cache[(key, lead)] = res
    return res