# CLAUDE.md

Project-level guidance for AI coding agents working in this repository.

## What this project is

**ECG-XAI-Visualizer** predicts cardiac ischemia from 12-lead ECG recordings with the
Cardiac-Ischemia-XNet model and explains each prediction with Grad-CAM, a saliency map,
SmoothGrad×SIGN and LIME, plus the share of the top-10 % attribution in P wave, QRS, ST
segment and T wave (NeuroKit2 delineation).

Input: MATLAB `.mat`, WFDB `.dat`/`.hea`, CSV, ASC, DICOM waveform `.dcm`, vendor XML, HL7 aECG.
Output: `xai_summary.csv` (50 columns, one row per analysed record; each method has its own
`<method>_*` share and `<method>_plot` columns), `<record>_xai.png` per record, optionally
`<record>_xai.npz` (with `all`: `<record>_xai_<method>.png`/`.npz`).
Optional pre-processing: notch (50/60 Hz), bandpass 0.5-40 Hz, wavelet baseline removal, EMD
denoise, resampling. Python 3.10+, MIT licensed.

## Essential commands

```bash
# setup (from repo root)
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# analyse a folder (Grad-CAM on lead II, one PNG per record)
python3 ecg_xai.py IN/ --output OUT/

# CSV/ASC need a rate; files without demographics need a CSV or fallbacks
python3 ecg_xai.py IN/ --output OUT/ --fs 500 --metadata ids.csv --age 60 --sex female \
    --pipeline notch bandpass --xai smoothgrad --xai_lead V5

# desktop GUI
python ui.py

# inspect inputs without the model (readers, lead mapping, age/sex resolution)
python3 debug_ecg.py IN/ --metadata ids.csv
```

## Programmatic usage (preferred for agents)

```python
import sys; sys.path.insert(0, "<repo_root>")
from runner.execution_runner import ExecutionRunner

runner = ExecutionRunner(
    config_path="<repo_root>/config.ini",   # defaults to repo config if omitted
    path_source="/any/input/file/or/folder",
    path_sink="/any/output/folder",
)
runner.default_fs = 500
runner.metadata_csv = "ids.csv"
runner.fallback_age, runner.fallback_sex = 60, "Female"
runner.xai_method = "gradcam"         # gradcam | saliency | smoothgrad | lime | all | none
runner.xai_lead = "II"
runner.keep_signals = True
records = runner.run()
runner.stats
```

Any input/output folders work. Do **not** copy user data into `data/input_ecgs`;
pass `path_source`/`path_sink` instead. Attributes not set keep their `config.ini` defaults.

## Verifying results

- `xai_summary.csv`: one row per *analysed* record, also with several methods (`all`); skipped
  and failed files get none (the log says why). Columns = `SUMMARY_COLUMNS` (50): `filename` ..
  `model`, then `xai_methods` (requested keys, comma-separated), `xai_lead`, `xai_target` (the
  first successful method's),
  then `<method>_p_wave`/`_qrs`/`_st_segment`/`_t_wave`/`_other` for every method in
  `XAI_METHODS` order, the shared `delineation` and `xai_note`, `patient_id` .. `notes`, and
  `<method>_plot` per method last (a new `XAI_METHODS` entry adds six columns). A method not run
  or failed has blank columns; a failure adds `<Method name>: explanation failed: <error>` to
  `xai_note`, which also holds the lead's delineation note and per-method share notes
  (`<Method name>: ...`). `runner.stats` counts records. Only written when at least one record
  succeeded (`runner.summary_path` is `None` otherwise); a stale file from an earlier run
  survives such a run. If it cannot be written (open in Excel), the rows go to
  `xai_summary_<date>-<time>.csv`, `runner.summary_error` says why and the CLI exits 1.
- `prediction` is `Ischemia` when `probability >= threshold`, compared before the CSV rounds
  `probability` to 6 decimals (`threshold` is written unrounded); `xai_target` equals
  `prediction` whenever an explanation ran.
- A method's five shares add up to 100 (± rounding); blank with an `xai_note` when the lead
  could not be segmented (flat/zero-filled lead, no R peaks; `delineation` blank too) or the
  method put no positive attribution in the lead's top percentile.
- `age_source`/`sex_source` say `file`, `csv` or `fallback`; `fs_source` says `file` or
  `fallback`.
- There is no test suite and no demo data in the repo (real ECGs are PHI). Validate changes on
  synthetic records: `neurokit2.ecg_simulate(..., method="ecgsyn")` gives a realistic lead; build
  a 12-lead from it (I, II, then III = II - I, aVR = -(I+II)/2, aVL = I - II/2, aVF = II - I/2,
  V1-V6 scaled copies) and write it in each format (`wfdb.wrsamp` with `comments=["Age: 71",
  "Sex: Male"]`, `scipy.io.savemat` with `val`/`fs`/`age`/`sex`, CSV with a lead-name header,
  ECG-Format-Converter's writers for DICOM/HL7). Cover: a rate-less CSV with 8 leads (I, II,
  V1-V6: derivation), a shuffled lead order (reordering), 250 and 1000 Hz (model-rate
  resampling), a record < 3 s, a 2-lead file (missing leads), and a broken file.
- Model parity: `ecgxai.inference.predict` and `ecgxai.explain.attribution_map` (gradcam,
  saliency) are bit-identical to the original beatsML functions given the same network and the
  same 500 Hz, standard-order input. Keep it that way; SmoothGrad matches with
  `smoothgrad_noise = 0` to float rounding (bit-identical with `_SMOOTHGRAD_BATCH = 1`).

## Behavior and gotchas

- **Model input contract** (`ecgxai/inference.py`, `ecgproc/standardize.py`): leads by *name*
  in FINAL_LEADS order (I, II, III, AVR, AVL, AVF, V1..V6), whole record resampled to the model
  rate (500 Hz) *before* the 3 s window is cut (`resample_signal`'s FFT path removes the
  end-to-start jump first; a ~20 ms residual remains where the model is most sensitive, so
  `window_start >= 0.1` s when results must not depend on the file's rate; a 500 Hz file
  resampled *up* with `resample` gets its model input from before that step), per-lead
  z-score with `std == 0 -> 1`, age `(age - 56.47) / 18.35`, sex Male = 1 / Female = 0. Every
  number comes from the `ModelSpec` in `ecgxai/registry.py`; never hard-code them elsewhere.
- `canonical_lead_name` strips `Lead`, `LEAD_`, `ECG`, `MDC_ECG_LEAD_` prefixes and anything
  after a comma or bracket (`Lead aVR, augmented voltage, right arm` -> `AVR`). It must never
  map a non-standard lead onto a standard one: `-aVR`, `Lead -aVR`, `MDC_ECG_LEAD_-aVR`, `V4R`,
  `MLII` stay unmatched (the separator after a prefix never swallows a minus sign), and so do
  names qualified as inverted or negative (`aVR (inverted)`).
- Missing leads skip the record unless `allow_missing_leads`; derivation (`derive_leads`, on by
  default) fills limb leads from two of I/II/III first. A channel with no finite sample or a
  constant value (`standardize.is_empty_channel`) counts as not recorded — checked on the *raw*
  signal, before filtering turns a constant into float noise. Non-numeric CSV columns become
  NaN channels (`analyze._as_float`) and fall under the same rule.
- **Order inside `analyze_file`**: read -> rate (`fs` or `default_fs`, else skip) -> age/sex
  (`ecgproc.patient.resolve`, before anonymizing, since the CSV may match the stored patient ID;
  missing -> skip) -> keep standard channels and interpolate NaN gaps (before any filter: a NaN
  turns a filtfilt output into all-NaN) -> pipeline -> resample -> standardize -> window ->
  predict -> explain each of `opts.xai_methods` (a failure stops only that method) ->
  figures/maps -> one row per record (methods side by side; `()`: XAI columns blank).
- XAI explains `prediction["class"]` (thresholded), not the arg-max. Grad-CAM registers a forward
  hook and removes it in `finally`; `LoadedModel.lock` serialises every forward/backward pass
  (the GUI's explain thread and a batch may share the cached model).
- SmoothGrad draws its noise per copy from a seeded CPU generator in a fixed order, so batching
  (`_SMOOTHGRAD_BATCH`) does not change the map beyond float rounding.
- LIME (`explain.lime_map`) switches each 50 ms window of each lead (`_lime_edges` from
  `spec.fs`; 12 x ceil(L / 25) features at 500 Hz) off with p = `_LIME_P_OFF` = 0.01 per copy
  (about 7 of 720), setting it to the lead's median, and z-scores every changed lead again, so
  it explains recorded window -> prediction (the gradient maps explain the z-scored
  input); working from `encode()`'s output is equivalent because the z-score ignores offset and
  scale and the median follows both. The `_LIME_*` constants are fixed on purpose: lime's own
  defaults (half the windows off, ridge alpha 1) give near-chance maps with 720 features. Row 0
  is the record as is; all masks are drawn in one call from a seeded torch CPU generator
  (`[xai] seed`, shared with SmoothGrad) before the first pass, so chunking cannot change them.
  Passes run in chunks of `_LIME_BATCH` = 10,
  each under `loaded.lock`, so `attribution_map` calls it outside its own lock. Log-odds of the
  explained class -> kernel-weighted ridge, penalty by GCV (`_ridge_gcv`) -> one weight per
  window, scaled with 0 kept in range (`_signed_unit`; `zero` = where 0 lands, rounded to
  float32 like the map so that `attr > zero` is never true for a weight of 0). Windows already
  at the off level are not features, so flat/zero-filled leads get exactly the zero level (no
  feature at all: zero map, no pass), and windows no copy switched off stay out of the fit with
  weight exactly 0 (fitted, their constant column leaves a rounding residue that counts as
  attribution when no weight is negative); a log-odds range under `_LIME_FLAT` (1e-6, relative)
  gives an all-zero map. `lime_samples` (default 1000, at least 2) is LIME's only own setting;
  `seed` is shared with SmoothGrad (`options()` checks it lies in -2**63 .. 2**64 - 1).
- **Delineation runs on the whole record** at the model rate and is cut to the window
  (`segments.analyze`): NeuroKit2's `dwt` raises for signals < 4 s, so delineating
  the 3 s window (what beatsML did) always fell back to fixed windows. `dwt`, then
  `prominence`, are tried on the lead as recorded and, only when its QRS points down (aVR, V1),
  also inverted (NeuroKit2 bounds almost nothing on a downward QRS; inverting an upright lead
  puts R on the S wave). A result counts only if it bounds QRS and T for at least half the
  beats (`dwt` returns T but no QRS above ~160 bpm); else fixed windows plus a note. `peak`
  returns no onset/offset pairs and `cwt` never helped on LUDB but takes minutes on long
  records: neither is used. The signed maps' zero level (SmoothGrad×SIGN ~0.5 after min-max,
  LIME where a weight of 0 lands; `attribution_map(..., with_zero=True)`, passed to `explain`
  as `zero`) is passed to `attribution_shares`, which weighs `attr - zero`; the [0, 1] map
  itself is unchanged (parity). Labels are painted
  P, T, ST, QRS so that later segments win overlaps; shares cannot exceed 100 %.
- A record's methods share one delineation per lead: `explain(..., delineations=)` caches
  `segments.delineate_safe` results (the tuple, or the failure reason as a `str`, never an
  exception object) by lead in `record["delineations"]`, which `explain_record` reuses; results
  equal each method run alone. The GUI runs `all` on config.ini's `[xai] lead` (II if unusable), and the viewer switches method and lead.
- Weights load with `weights_only=True`; numpy's scalar/dtype stay allow-listed under the
  numpy<2 name `numpy.core.multiarray.scalar` for checkpoints that carry metrics (as the beatsML
  one did; see `_read_checkpoint`; the submodule is imported, since numpy 1.26.4's
  `numpy._core` stub does not load it). Requires torch >= 2.6 (`_read_checkpoint` and
  `check_dependencies` say so).
- `models/finetuned_ischemia_robust_model.pth` is the bare state_dict of an SWA `AveragedModel`
  (`module.` prefix plus `n_averaged`; `registry._state_dict` unwraps it the way the fine-tuning
  code does) and stores no metadata. `ecgxai/architectures.py` is the fine-tuning code's network
  verbatim and `ModelSpec.arch` its constructor arguments (`num_layers=3`, `width_scale=1.0`,
  `drop_path_rate=0.2`, `dropout_rate=0.5`, ReLU); logits and Grad-CAM maps are bit-identical
  to the user's Breisgau PCI inference script. Do not change either to match the beatsML
  network (five layers, leaky ReLU). The 0.564 threshold is the beatsML app's deployed value,
  kept for this model by the user's decision (that script labels at 0.5). Importing `ecgxai.registry`, `ecgxai.explain` or `runner` must not
  import torch (GUI start-up time); torch is imported inside functions.
- `ecgproc/readers.py`, `preprocess.py`, `leads.py`, `files.py`, `xml_utils.py` are shared with
  ECG-Format-Converter 1.2.0. Local divergences, to port back: `read_dat_hea` returns header
  `comments`; `read_hl7_aecg` reads `<birthTime><age .../>`; `_denoise_lead` thresholds nothing
  when no IMF lies above `noise_cut` (1.2.0 fell back to IMF 1, erasing clean/bandpassed QRS);
  `find_files` drops a `.mat` that has a same-stem `.hea` (PhysioNet Challenge records were
  read twice), maps a single such `.mat` to its `.hea`, and skips macOS `._*` / `__MACOSX`
  sidecars; `resample_signal` (fft) removes the end-to-start line before resampling;
  `_filtfilt` pads by up to 3 s (fs passed by notch/bandpass); `wavelet_baseline_removal`
  rebuilds the approximation with `waverec` (upcoef(take=n) lagged up to 0.4 s), takes `fs`
  to scale `level` (given at 500 Hz; the pipeline uses 9 = 0-0.49 Hz); `emd_denoise` raises
  when every lead failed and says 'kept whole' for threshold 0;
  `merge_emd_options` validates `noise_width`/`max_imf`/`seed` and keeps an explicit `None`;
  `read_csv`/`read_asc` read `;`, tab and space separated files, decimal commas and a BOM;
  `read_xml`/`read_hl7_aecg` invent no 500 Hz rate (fs `None` -> `--fs`), take the rate of the
  rhythm block, accept '1000 Hz', pad leads to the longest and list them in
  `meta["padded_leads"]` (analyze cuts the record to the common length); `parse_samples` splits
  on any whitespace, `,` or `;`; upper-case `REC.HEA`/`REC.DAT` are read on case-sensitive file systems through
  temporary links; clearer errors for MATLAB v7.3 and ECG-Format-Converter's generic XML.
- Sex parsing accepts text only (`ecgproc.patient.parse_sex`); numeric codes are ambiguous
  across datasets and stay unknown. Dates are parsed only when unambiguous.
- Metadata CSV (`MetadataTable`): record-level ID columns match file names, patient-level ones
  the stored patient ID, a plain `ID` both — never let a file number match a patient column.
  File names are tried most specific first: the record id (`sub/rec.csv`, WFDB as `rec.hea`,
  passed as `resolve(..., record_id=)`), then the file's name and stem. Read as UTF-8, else
  cp1252 (Excel); `Age (years)` headers count as `age`; a repeated ID keeps earlier values
  for blank cells. With `anonymize`, `resolve(..., reveal_ids=False)` keeps the matched ID out
  of `*_detail`, and patient ID, names, birth date and exam date/time are blanked.
- `ExecutionRunner.options()` validates everything (pipeline steps, methods, lead, threshold,
  fallbacks, CSV path, EMD options, PyEMD present for `emd`) and raises `ValueError` before any
  file is touched; keep new settings validated there. `_load_config` turns configparser errors
  (BOM, `%`, duplicates, no section) into `ValueError`, matches section names case-insensitively
  and rejects unknown sections and keys (`_CONFIG_KEYS`: add new settings there).
- Record ids are paths relative to the input folder (WFDB pairs as `.hea`); output bases flatten
  subfolders with `__`, add `_<ext>` when two inputs share a stem, and a counter if names still
  collide (compared case-insensitively); flattened names over 180 bytes keep their end plus a
  hash. Files are `<base>_xai.png`/`.npz` for one requested method, `<base>_xai_<method>.png`/
  `.npz` for several (by the number requested, not succeeded); a row's `<method>_plot` columns
  name the PNGs. `input_files()` leaves out an output folder nested in the input, summary files
  (`is_summary_file`: `xai_summary.csv` and the time-stamped fallback) and the metadata CSV. The
  output folder must be writable before the model loads. The summary is written in a
  `finally`, so Ctrl-C keeps the finished rows; an explicit `config_path` must exist.
- The GUI never kills a running batch: closing the window cancels it and closes once the
  current file is done (`_close_requested`); pickers and drops are ignored while it runs. A
  folder walk (`ScanWorker`) cannot be interrupted either: closing hides the window and the app
  quits when the walk ends, and `main()` waits for running scans after `app.exec()` (Cmd+Q).
  The window applies `ExecutionRunner.filter_inputs()` to the scanned list itself, and Start
  waits until the scan of the current input is done.
- `ecg_xai.py` runs `ecgproc.deps.check_dependencies()` before importing `ecgxai.cli` (which
  pulls in numpy, pandas, tqdm); `ui.py` runs it with `import_modules=False` (looks packages up
  without importing torch). lxml and EMD-signal are optional there: missing, they only print a
  note, and `options()` rejects an `emd` step PyEMD cannot run. `emd_denoise` raises when every
  lead failed (old EMD-signal on NumPy 2) instead of returning the record unfiltered.
- GUI code paths must stay callback-based (`progress_cb`/`log_cb`/`cancel_cb`); no prints or
  tqdm inside `runner/` or `ecgxai/analyze.py` except through the `log` parameter. Plotting goes
  through `ecgxai.plotting` on a Figure (never pyplot), so worker threads can render PNGs.

## Code map

```
runner/execution_runner.py   orchestrator: config, validation, discover -> analyse -> summary
ecgxai/analyze.py            per-file pipeline; returns {"status", "file", "reason", "row",
                             "record", "outputs"}, one row per record (methods side by side)
ecgxai/registry.py           ModelSpec catalogue, device, safe checkpoint loading (cached)
ecgxai/architectures.py      Cardiac-Ischemia-XNet (attribute names = checkpoint keys)
ecgxai/inference.py          model-rate resampling, window, z-score, predict
ecgxai/explain.py            Grad-CAM, saliency, SmoothGrad x SIGN, LIME (lime_map), explain()
ecgxai/segments.py           NeuroKit2 delineation + attribution shares
ecgxai/plotting.py, theme.py figures for the viewer and the PNG exports
ecgxai/cli.py                argparse + interactive prompts, delegates to ExecutionRunner
ecgproc/                     readers, filters, lead derivation, lead-name standardization,
                             age/sex resolution
ui.py                        PySide6 app (ScanWorker, ProcessWorker, ExplainWorker, viewer)
skills/ecg-xai-visualizer/   canonical agent skill; skills/sync.sh copies it to
                             .claude/skills/ (Claude Code) and .opencode/skills/ (opencode)
.github/workflows/build.yml  PyInstaller builds (Linux x64, Win x64, macOS arm64), CPU torch
```

## Release flow (maintainers)

1. Merge to `main`, bump `__version__` in `ecgxai/__init__.py` (the GUI header reads it
   from there) and `version:` in `CITATION.cff` if needed.
2. Either trigger *Actions -> Build Executables* manually (artifacts only), or push a `v*` tag:
   the workflow then builds all three zips and attaches them to a GitHub release automatically.
