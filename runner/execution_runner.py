import configparser
import hashlib
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from ecgproc.constants import SUPPORTED_EXTENSIONS
from ecgproc.files import find_files
from ecgproc.patient import load_metadata_csv, parse_age, parse_sex
from ecgproc.preprocess import EMD_DEFAULTS, PIPELINE_STEPS, merge_emd_options
from ecgxai.explain import XAI_METHODS
from ecgxai.registry import DEFAULT_MODEL, app_root, get_spec

SHARE_SUFFIXES = ("p_wave", "qrs", "st_segment", "t_wave", "other")
SUMMARY_COLUMNS = [
    "filename",
    "prediction",
    "probability",
    "threshold",
    "model",
    "xai_methods",
    "xai_lead",
    "xai_target",
    *(f"{m}_{s}" for m in XAI_METHODS for s in SHARE_SUFFIXES),
    "delineation",
    "xai_note",
    "patient_id",
    "patient_age",
    "age_source",
    "patient_sex",
    "sex_source",
    "exam_date",
    "exam_time",
    "sampling_freq",
    "fs_source",
    "n_samples",
    "original_leads",
    "calculated_leads",
    "missing_leads",
    "pipeline",
    "window_start_s",
    "notes",
    *(f"{m}_plot" for m in XAI_METHODS),
]
SUMMARY_NAME = "xai_summary.csv"
_SUMMARY_FILE = re.compile(r"xai_summary(_\d{8}-\d{6})?\.csv", re.IGNORECASE)


def is_summary_file(name):
    """True for a summary this tool wrote (never an ECG input)."""
    return bool(_SUMMARY_FILE.fullmatch(str(name)))
XAI_CHOICES = ("gradcam", "saliency", "smoothgrad", "lime", "all", "none")


def default_config_path():
    return app_root() / "config.ini"


class ExecutionRunner:

    def __init__(self, config_path=None, path_source=None, path_sink=None, use_config=True):
        root = app_root()
        self.path_source = Path(path_source) if path_source else root / "data" / "input_ecgs"
        self.path_sink = Path(path_sink) if path_sink else root / "data" / "results"

        self.pipeline = []                  # subset of notch, bandpass, wavelet, emd (in order)
        self.notch = 50                     # 50 or 60 Hz
        self.default_fs = None              # rate to assume for files that store none
        self.resample = None                # resample the processed signal (Hz) or None
        self.resample_method = "fft"        # fft | interpolation (also for the model input)
        self.derive_leads = True            # Einthoven / Goldberger for missing limb leads
        self.anonymize = False
        self.emd_options = dict(EMD_DEFAULTS)
        # [model]
        self.model = DEFAULT_MODEL
        self.threshold = None               # None = the model's own (ModelSpec.threshold)
        self.window_start = 0.0             # seconds
        self.allow_missing_leads = False    # True: zero-fill instead of skipping
        self.device = "cpu"                 # cpu | cuda | mps | auto
        # [patient]
        self.metadata_csv = None            # ID / age / sex table
        self.fallback_age = None            # years
        self.fallback_sex = None            # "Male" | "Female"
        # [xai]
        self.xai_method = "gradcam"         # gradcam | saliency | smoothgrad | lime | all | none
        self.xai_lead = "II"
        self.top_percentile = 90.0
        self.save_plots = True              # <record>_xai.png (all: <record>_xai_<method>.png)
        self.save_maps = False              # <record>_xai.npz (all: <record>_xai_<method>.npz)
        self.smoothgrad_samples = 50
        self.smoothgrad_noise = 0.2
        self.lime_samples = 1000            # LIME: perturbed copies (model passes) per record
        self.seed = 0                       # None = fresh SmoothGrad noise and LIME masks every run
        self.keep_signals = False
        self.max_kept_signals = 500

        # Populated by run()
        self.summary = []
        self.summary_path = None
        self.summary_error = None
        self.stats = {"success": 0, "skipped": 0, "failed": 0, "positive": 0}
        self.cancelled = False
        self.model_description = ""

        if not use_config:
            return
        if config_path and not Path(config_path).is_file():
            raise ValueError(f"Config file not found: {config_path}")
        self._load_config(Path(config_path) if config_path else default_config_path())

    # config

    def _load_config(self, config_path):
        if not config_path or not config_path.is_file():
            return
        cfg = configparser.ConfigParser(interpolation=None)
        try:
            with open(config_path, encoding="utf-8-sig") as fh:
                cfg.read_file(fh)
        except UnicodeError as e:
            raise ValueError(f"{config_path}: {e} (save the file as UTF-8)") from None
        except (configparser.Error, OSError) as e:
            raise ValueError(f"{config_path}: {e}") from None
        sections = {}
        for name in cfg.sections():
            if name.lower() in sections:
                raise ValueError(f"{config_path}: section [{name}] appears twice.")
            sections[name.lower()] = name
        unknown = [f"[{name}]" for low, name in sections.items() if low not in _CONFIG_KEYS]
        defaults = set(cfg.defaults())
        everywhere = set(EMD_DEFAULTS).union(*(keys for keys in _CONFIG_KEYS.values() if keys))
        unknown += [f"[DEFAULT] {key}" for key in defaults if key not in everywhere]
        for low, name in sections.items():
            if low in _CONFIG_KEYS:
                known = _CONFIG_KEYS[low] or set(EMD_DEFAULTS)
                unknown += [f"[{name}] {key}" for key in cfg[name] if key not in known and key not in defaults]
        if unknown:
            raise ValueError(f"{config_path}: unknown setting(s) {', '.join(unknown)}.")

        def section(name):
            return cfg[sections[name]] if name in sections else {}

        def text(sec, key):
            return (sec.get(key, "") or "").strip()

        def num(sec_name, key, kind, default):
            raw = text(section(sec_name), key)
            if not raw:
                return default
            try:
                return kind(raw)
            except ValueError:
                raise ValueError(f"{config_path}: [{sec_name}] {key} must be a number, "
                                 f"got '{raw}'.") from None

        def flag(sec_name, key, default):
            raw = text(section(sec_name), key).lower()
            if not raw:
                return default
            if raw in ("1", "yes", "true", "on"):
                return True
            if raw in ("0", "no", "false", "off"):
                return False
            raise ValueError(f"{config_path}: [{sec_name}] {key} must be true or false, got '{raw}'.")

        proc = section("processing")
        steps = text(proc, "pipeline")
        if steps:
            self.pipeline = [p.strip().lower() for p in steps.split(",") if p.strip()]
        self.notch = num("processing", "notch", int, self.notch)
        self.default_fs = num("processing", "default_fs", float, None)
        self.resample = num("processing", "resample", float, None)
        self.resample_method = text(proc, "resample_method").lower() or self.resample_method
        self.derive_leads = flag("processing", "derive_leads", self.derive_leads)
        self.anonymize = flag("processing", "anonymize", self.anonymize)

        mdl = section("model")
        self.model = text(mdl, "name") or self.model
        self.threshold = num("model", "threshold", float, None)
        self.window_start = num("model", "window_start", float, self.window_start)
        self.allow_missing_leads = flag("model", "allow_missing_leads", self.allow_missing_leads)
        self.device = text(mdl, "device").lower() or self.device

        pat = section("patient")
        self.metadata_csv = text(pat, "metadata_csv") or None
        self.fallback_age = text(pat, "fallback_age") or None
        self.fallback_sex = text(pat, "fallback_sex") or None

        xai = section("xai")
        self.xai_method = text(xai, "method").lower() or self.xai_method
        self.xai_lead = text(xai, "lead") or self.xai_lead
        self.top_percentile = num("xai", "top_percentile", float, self.top_percentile)
        self.save_plots = flag("xai", "save_plots", self.save_plots)
        self.save_maps = flag("xai", "save_maps", self.save_maps)
        self.smoothgrad_samples = num("xai", "smoothgrad_samples", int, self.smoothgrad_samples)
        self.smoothgrad_noise = num("xai", "smoothgrad_noise", float, self.smoothgrad_noise)
        self.lime_samples = num("xai", "lime_samples", int, self.lime_samples)
        if "xai" in sections and "seed" in section("xai"):
            self.seed = num("xai", "seed", int, None)

        if "emd" in sections:
            self._load_emd_config(section("emd"), config_path)

    def _load_emd_config(self, sec, config_path):
        for key, default in EMD_DEFAULTS.items():
            raw = sec.get(key, "").strip()
            if not raw:
                continue
            try:
                if isinstance(default, bool):
                    value = sec.getboolean(key)
                elif isinstance(default, float):
                    value = float(raw)
                elif isinstance(default, int) or key in ("max_imf", "seed"):
                    value = int(raw)
                else:
                    value = raw.lower()
            except ValueError:
                raise ValueError(f"{config_path}: [emd] {key} has invalid value '{raw}'.") from None
            self.emd_options[key] = value

    # validation

    def options(self):
        from ecgxai.analyze import AnalysisOptions
        from ecgxai.explain import lead_key, method_key

        unknown = [s for s in self.pipeline if s not in PIPELINE_STEPS]
        if unknown:
            raise ValueError(f"Unknown pipeline step(s): {', '.join(unknown)}. "
                             f"Choose from {', '.join(PIPELINE_STEPS)}.")
        if self.resample_method not in ("fft", "interpolation"):
            raise ValueError(f"Unknown resampling method '{self.resample_method}'. "
                             "Choose from 'fft' or 'interpolation'.")
        if int(self.notch) not in (50, 60):
            raise ValueError(f"Unknown notch frequency '{self.notch}'. Choose 50 or 60 Hz.")
        for key, val in (("default_fs", self.default_fs), ("resample", self.resample)):
            if val is not None and (not _finite(val) or float(val) <= 0):
                raise ValueError(f"{key} must be a positive number of Hz, got '{val}'.")
        spec = get_spec(self.model)
        if self.threshold is not None and not (0.0 < float(self.threshold) < 1.0):
            raise ValueError(f"threshold must lie between 0 and 1, got '{self.threshold}'.")
        if not _finite(self.window_start) or float(self.window_start) < 0:
            raise ValueError(f"window_start must be 0 or more seconds, got '{self.window_start}'.")
        if str(self.device).lower() not in ("cpu", "cuda", "mps", "auto"):
            raise ValueError(f"Unknown device '{self.device}'. Choose from cpu, cuda, mps, auto.")

        method = str(self.xai_method or "none").strip().lower()
        try:
            methods = (() if method in ("", "none", "off") else tuple(XAI_METHODS) if method == "all"
                       else (method_key(method),))
        except ValueError:
            raise ValueError(f"Unknown XAI method '{self.xai_method}'. Choose from "
                             f"{', '.join(XAI_CHOICES[:-1])} or {XAI_CHOICES[-1]}.") from None
        lead = lead_key(self.xai_lead)
        if not (0.0 < float(self.top_percentile) < 100.0):
            raise ValueError(f"top_percentile must lie between 0 and 100, got '{self.top_percentile}'.")
        if int(self.smoothgrad_samples) < 1:
            raise ValueError("smoothgrad_samples must be at least 1.")
        if not _finite(self.smoothgrad_noise) or float(self.smoothgrad_noise) < 0:
            raise ValueError(f"smoothgrad_noise must be a number of 0 or more, got '{self.smoothgrad_noise}'.")
        lime_samples = float(self.lime_samples) if _finite(self.lime_samples) else math.nan
        if not (lime_samples >= 2 and lime_samples == int(lime_samples)):
            raise ValueError(f"lime_samples must be a whole number of at least 2 (1000 or more "
                             f"recommended), got '{self.lime_samples}'.")
        try:
            seed_ok = self.seed is None or -2 ** 63 <= int(self.seed) < 2 ** 64
        except (TypeError, ValueError, OverflowError):
            seed_ok = False
        if not seed_ok:
            raise ValueError(f"seed must be a whole number from -2**63 to 2**64 - 1, or blank for new "
                             f"draws every run, got '{self.seed}'.")
        if self.max_kept_signals is not None and (not _finite(self.max_kept_signals)
                                                  or int(self.max_kept_signals) < 0):
            raise ValueError(f"max_kept_signals must be a whole number of 0 or more (or None for no "
                             f"limit), got '{self.max_kept_signals}'.")

        age = None
        if self.fallback_age not in (None, ""):
            age = parse_age(self.fallback_age)
            if age is None:
                raise ValueError(f"fallback_age must be an age in years (0-125), got '{self.fallback_age}'.")
        sex = None
        if self.fallback_sex not in (None, ""):
            sex = parse_sex(self.fallback_sex)
            if sex is None:
                raise ValueError(f"fallback_sex must be Male or Female (M/F), got '{self.fallback_sex}'.")
        if self.metadata_csv and not Path(self.metadata_csv).is_file():
            raise ValueError(f"Metadata CSV not found: {self.metadata_csv}")

        if "emd" in self.pipeline:
            from ecgproc import preprocess
            if not preprocess._PYEMD_AVAILABLE:
                raise ValueError("The 'emd' step needs PyEMD, which cannot be imported. Install it with "
                                 "'pip install EMD-signal', or leave 'emd' out of the pipeline.")
        emd = merge_emd_options(self.emd_options)
        return AnalysisOptions(
            pipeline=list(self.pipeline),
            notch=int(self.notch),
            default_fs=float(self.default_fs) if self.default_fs else None,
            resample=float(self.resample) if self.resample else None,
            resample_method=self.resample_method,
            derive_leads=bool(self.derive_leads),
            anonymize=bool(self.anonymize),
            emd_options=emd,
            threshold=float(self.threshold) if self.threshold is not None else spec.threshold,
            window_start=float(self.window_start),
            allow_missing_leads=bool(self.allow_missing_leads),
            fallback_age=age,
            fallback_sex=sex,
            xai_methods=methods,
            xai_lead=lead,
            xai_options={
                "smoothgrad_samples": int(self.smoothgrad_samples),
                "smoothgrad_noise": float(self.smoothgrad_noise),
                "lime_samples": int(lime_samples),
                "seed": None if self.seed is None else int(self.seed),
                "top_percentile": float(self.top_percentile),
            },
            save_plots=bool(self.save_plots),
            save_maps=bool(self.save_maps),
        )

    # run

    def record_names(self, files):
        src = Path(self.path_source)
        root = src if src.is_dir() else src.parent
        ids, stems = [], []
        for f in files:
            try:
                rel = f.relative_to(root)
            except ValueError:
                rel = Path(f.name)
            if f.suffix.lower() in (".dat", ".hea"):
                rel = rel.with_suffix(".hea")
            ids.append(rel.as_posix())
            stem = rel.with_suffix("").as_posix().replace("/", "__")
            if len(stem.encode()) > 180:
                digest = hashlib.sha1(stem.encode()).hexdigest()[:8]
                stem = f"{stem.encode()[-150:].decode(errors='ignore').lstrip('_')}__{digest}"
            stems.append(stem)
        counts = {}
        for s in stems:
            counts[s.lower()] = counts.get(s.lower(), 0) + 1
        bases, taken = [], set()
        for s, i in zip(stems, ids):
            base = f"{s}_{Path(i).suffix.lstrip('.').lower()}" if counts[s.lower()] > 1 else s
            unique, n = base, 2
            while unique.lower() in taken:          # same.csv + same.CSV, a/b + a__b
                unique, n = f"{base}_{n}", n + 1
            taken.add(unique.lower())
            bases.append(unique)
        return ids, bases

    def input_files(self):
        return self.filter_inputs(find_files(self.path_source, SUPPORTED_EXTENSIONS))

    def filter_inputs(self, files):
        given = Path(self.path_source)
        src, sink = given.resolve(), Path(self.path_sink).resolve()
        nested = ()
        if src.is_dir() and sink != src and src in sink.parents:
            nested = (os.path.join(str(given), str(sink.relative_to(src))) + os.sep,
                      str(sink) + os.sep)
        meta = Path(self.metadata_csv).resolve() if self.metadata_csv else None
        kept = []
        for f in files:
            if is_summary_file(f.name):
                continue
            if meta is not None and f.name == meta.name and f.resolve() == meta:
                continue
            if nested and str(f).startswith(nested):
                continue
            kept.append(f)
        return kept

    def run(self, progress_cb=None, log_cb=None, cancel_cb=None):
        from ecgxai.analyze import analyze_file
        from ecgxai.registry import describe, load_model

        log = log_cb if log_cb is not None else print
        self.path_source, self.path_sink = Path(self.path_source), Path(self.path_sink)
        self.summary = []
        self.summary_path = None
        self.summary_error = None
        self.cancelled = False
        self.stats = {"success": 0, "skipped": 0, "failed": 0, "positive": 0}
        records = {}

        files = self.input_files()
        if not files:
            raise ValueError(f"No ECG files found in {self.path_source}")
        opts = self.options()
        if self.path_sink.exists() and not self.path_sink.is_dir():
            raise ValueError(f"Output folder {self.path_sink} is a file.")
        try:
            self.path_sink.mkdir(parents=True, exist_ok=True)
            tempfile.TemporaryFile(dir=self.path_sink).close()      # writable? (before the batch)
        except OSError as e:
            raise ValueError(f"Cannot write to the output folder {self.path_sink}: "
                             f"{e.strerror or e}") from None

        loaded = load_model(self.model, self.device)
        self.model_description = describe(loaded)
        log(f"Model: {self.model_description}")
        spec = loaded.spec

        table = None
        if self.metadata_csv:
            table = load_metadata_csv(self.metadata_csv, log=log)

        ids, bases = self.record_names(files)
        total = len(files)
        positive_label = spec.label(spec.positive_class)

        try:
            for i, file in enumerate(files):
                if cancel_cb is not None and cancel_cb():
                    self.cancelled = True
                    log(f"Cancelled after {i} of {total} files.")
                    break
                if progress_cb is not None:
                    progress_cb(i, total, ids[i])
                keep = self.keep_signals and (self.max_kept_signals is None
                                              or len(records) < int(self.max_kept_signals))
                try:
                    res = analyze_file(file, ids[i], bases[i], opts, loaded, self.path_sink,
                                       patient_table=table, log=log, keep=keep)
                except Exception as e:   # analyze_file reports its own errors; last resort
                    log(f"Error processing {ids[i]}: {e}")
                    res = {"status": "failed", "row": None, "record": None}
                self.stats[res["status"]] += 1
                if res["status"] == "success":
                    self.summary.append(res["row"])
                    if res["row"]["prediction"] == positive_label:
                        self.stats["positive"] += 1
                    if res["record"] is not None:
                        records[ids[i]] = res["record"]
        finally:
            self._write_summary(log)

        if progress_cb is not None and not self.cancelled:
            progress_cb(total, total, "")
        if self.keep_signals and self.stats["success"] > len(records):
            log(f"The viewer keeps the first {len(records)} records; "
                f"{SUMMARY_NAME} lists all {self.stats['success']}.")
        return records

    # summary

    def _write_summary(self, log=print):
        if not self.summary:
            self.summary_path = None
            return
        df = pd.DataFrame(self.summary)
        for col in SUMMARY_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        target = self.path_sink / SUMMARY_NAME
        try:
            df[SUMMARY_COLUMNS].to_csv(target, index=False)
            self.summary_path = target
            return
        except OSError as e:
            self.summary_error = f"Could not write {target}: {e.strerror or e}"
        alt = self.path_sink / f"{target.stem}_{datetime.now():%Y%m%d-%H%M%S}.csv"
        try:
            df[SUMMARY_COLUMNS].to_csv(alt, index=False)
            self.summary_path = alt
            log(f"{self.summary_error} (is it open in another program?). The results were written "
                f"to {alt.name} instead.")
        except OSError as e:
            self.summary_path = None
            log(f"{self.summary_error}; {alt.name} could not be written either: {e}")

    # bootstrap

    @classmethod
    def bootstrap(cls):
        runner = cls()
        records = runner.run()
        print("\n### Summary ###")
        print(f"Analysed: {runner.stats['success']} "
              f"({runner.stats['positive']} predicted positive)")
        print(f"Skipped: {runner.stats['skipped']}")
        print(f"Failed: {runner.stats['failed']}")
        if runner.summary_path:
            print(f"Results: {runner.summary_path}")
        return records


# Settings
_CONFIG_KEYS = {
    "processing": {"pipeline", "notch", "default_fs", "resample", "resample_method", "derive_leads",
                   "anonymize"},
    "model": {"name", "threshold", "window_start", "allow_missing_leads", "device"},
    "patient": {"metadata_csv", "fallback_age", "fallback_sex"},
    "xai": {"method", "lead", "top_percentile", "save_plots", "save_maps", "smoothgrad_samples",
            "smoothgrad_noise", "lime_samples", "seed"},
    "emd": None,
}


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False
