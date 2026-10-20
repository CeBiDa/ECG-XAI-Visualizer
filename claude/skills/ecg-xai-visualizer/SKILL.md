---
name: ecg-xai-visualizer
description: Predict cardiac ischemia from 12-lead ECG recordings (MAT, WFDB, CSV, ASC, DICOM, vendor XML, HL7 aECG) with the Cardiac-Ischemia-XNet model and explain each prediction with Grad-CAM, saliency maps, SmoothGrad×SIGN or LIME, including the share of attribution in the P wave, QRS, ST segment and T wave; writes xai_summary.csv and one figure per record and method. Use when the user has ECG files and wants ischemia predictions, model explanations or attribution figures. Triggers on "ischemia prediction", "explain ECG model", "Grad-CAM ECG", "saliency ECG", "SmoothGrad", "LIME ECG", "XAI ECG", "Cardiac-Ischemia-XNet", "beatsML".
license: MIT
---

# ECG-XAI-Visualizer skill

Drive the ECG-XAI-Visualizer CLI/Python API to run the Cardiac-Ischemia-XNet model on ECG files and explain its predictions. Works on single files or whole folders with mixed formats.

## When to use

- User supplies ECG signal files (.mat, .dat/.hea, .csv, .asc, .dcm, .xml, .hl7) and wants an
  ischemia prediction, an explanation of it (Grad-CAM, saliency, SmoothGrad×SIGN, LIME),
  attribution figures, or a table of predictions and P/QRS/ST/T attribution shares.

## Setup

```bash
git clone https://github.com/CeBiDa/ECG-XAI-Visualizer
cd ECG-XAI-Visualizer
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional
pip install -r requirements.txt
```

## Recommended: Python API

```python
import sys
sys.path.insert(0, "/path/to/ECG-XAI-Visualizer")   # repo root

from runner.execution_runner import ExecutionRunner

runner = ExecutionRunner(
    config_path="/path/to/ECG-XAI-Visualizer/config.ini",
    path_source="/abs/input",
    path_sink="/abs/output",
)
runner.default_fs = 500
runner.metadata_csv = "/abs/ids.csv"
runner.fallback_age = 60
runner.fallback_sex = "Female"
runner.pipeline = ["notch", "bandpass"]   # optional
runner.xai_method = "gradcam"             # gradcam | saliency | smoothgrad | lime | all | none
runner.xai_lead = "II"
runner.keep_signals = True
records = runner.run()
print(runner.stats, runner.summary_path)
```

- `records` is `{record_id: {"prediction": {...}, "xai": {(method, lead): {...}}, "delineations":
  {lead: ...}, "patient": {...}, "raw": {...}, "processed": {...}, ...}}` (`"xai"` has one entry
  per method that succeeded, `"delineations"` the lead segmentation they share), populated only
  when `keep_signals = True` and capped at `max_kept_signals` (500).
- `ecgxai.analyze.explain_record(ecgxai.registry.load_model(), records[rid], "saliency", "V5")`
  explains a kept record with another method or lead without re-running the batch, reusing the
  method's map and the lead's delineation when the record has them.
- `run()` **raises ValueError** if no supported file is found or a setting is invalid (unknown
  pipeline step, method, lead, threshold outside 0-1, unparseable fallback, missing CSV). It
  does not return empty.

## CLI recipes

```bash
python3 ecg_xai.py IN/ --output OUT/

python3 ecg_xai.py IN/ --output OUT/ --fs 500 --pipeline notch bandpass \
    --xai smoothgrad --xai_lead V5

python3 ecg_xai.py IN/ --output OUT/ --xai all

python3 ecg_xai.py IN/ --output OUT/ --metadata ids.csv --age 60 --sex female

python3 ecg_xai.py IN/ --output OUT/ --xai none --anonymize
```

## Input rules that break naive checks

- **A record needs a sampling rate, an age and a sex, or it is skipped** (logged, counted in
  `skipped`, no summary row). CSV/ASC never carry any of the three: pass `--fs` and a metadata
  CSV or `--age`/`--sex`. HL7/XML files without a usable rate and MAT files without an `fs`
  variable also need `--fs`. Age/sex
  sources, per field: file -> CSV -> fallback.
- **Metadata CSV matching**: a plain `ID` column matches the record id (`sub/rec.csv`, WFDB as
  `rec.hea`) or the file name (with or without extension), the patient ID stored in the file,
  or the leading number of the file name;
  `record_id`/`ecg_id`/`filename`/`study_id` columns match file names only; `patient_id`/
  `subject_id` columns match the stored patient ID only. The run log names the columns used.
  Sex must be text (M/F, male/female, w/weiblich); numeric codes are ignored.
- **Leads are matched by name** (case-insensitive; `Lead`, `MDC_ECG_LEAD_` prefixes stripped).
  Missing limb leads are derived from two of I/II/III; a record with leads still missing is
  skipped unless `--allow-missing-leads` (then zero-filled, out of distribution). `MLII`,
  `V4R`, `-aVR`, `Lead -aVR` are not standard leads. Channels with no samples or a flat line count as not
  recorded.
- **Input discovery recurses.** Record ids (`filename` column) are paths relative to the input
  folder; figures are `<id without its extension, / -> __>_xai.png`, and with `--xai all`
  `<...>_xai_<method>.png` per method (`gradcam`, `saliency`, `smoothgrad`, `lime`); take the
  exact names from the row's `<method>_plot` columns. WFDB pairs count once, as `<stem>.hea`.
- **Exit codes**: 2 = invalid setting, missing `--config` file, unusable output folder, or a
  model that cannot be loaded; 1 = no input file found,
  failures with zero successes, or `xai_summary.csv` not writable (rows then in
  `xai_summary_<date>-<time>.csv`, see `runner.summary_path`/`summary_error`); 0 otherwise. A batch with skips or some failures exits 0 —
  read `runner.stats` or the `### Summary ###` block.
- An output folder inside the input folder, `xai_summary.csv` files and the metadata CSV are not
  analysed. A PhysioNet Challenge record (`.hea` + `.mat` signal file) counts once, as `.hea`.

## Verify

- `OUT/xai_summary.csv` has one row per **analysed** record, also with `--xai all` (50
  columns): `xai_methods` lists the methods requested (`gradcam,saliency,smoothgrad,lime` for
  `all`), and every method has its own share columns (`gradcam_qrs`, `lime_st_segment`, ...)
  and `<method>_plot`, blank when it did not run. A method that failed leaves its columns blank
  and `xai_note` says `<Method name>: explanation failed: ...`; the other methods' results stay.
  `runner.stats` counts records. A stale summary from an earlier run survives a run
  in which nothing succeeded: use `runner.summary_path` (`None` = nothing succeeded) or a fresh
  output folder.
- `prediction == "Ischemia"` when `probability >= threshold` (default 0.564); the comparison
  uses the unrounded probability (the CSV holds 6 decimals). `xai_target == prediction` whenever
  an explanation ran (blank with `--xai none`, or when every requested method failed).
- For each method `m` that ran, `m_p_wave + m_qrs + m_st_segment + m_t_wave + m_other ≈ 100`
  when `delineation` is set; `delineation == "window"` means no NeuroKit2 delineator bounded the
  QRS and T of most beats (`xai_note` says which failed) and fixed windows around R peaks were
  used (records under 4 s usually get `prominence`); blank shares come with an `xai_note`.
- Check `age_source`/`sex_source` (`file`, `csv`, `fallback`) and `fs_source` before
  prediction; `calculated_leads` were derived, `missing_leads` were zero-filled.
- `.npz` (with `--save-maps`; one `_xai_<method>.npz` per method with `--xai all`):
  `model_input` (before the per-lead z-score) and `attribution` are (12, 1500) at 500 Hz,
  leads I..V6 (fewer samples for records shorter than 3 s); `segment_labels` (0 = P, 1 = QRS,
  2 = ST, 3 = T, -1 = other) belong to the explained `lead` (shares use the samples at or above
  `top_percentile`); `zero_level` is where no attribution lies in the map (0, or the zero of a
  signed map: ~0.5 for SmoothGrad×SIGN, wherever a weight of 0 lands for LIME, whose map holds
  one value per lead and 50 ms); `xai_note` holds the lead's delineation note and that
  method's share note.

## Troubleshooting

- A record was skipped: the log line says why. `python3 debug_ecg.py <file> [--metadata ids.csv]`
  shows the rate, how each lead name maps, and which age/sex the model would get (and from where).
- "unknown sampling rate": add `--fs <Hz>`.
- "no age" / "no sex": the file has none and the CSV did not match — check the ID column against
  the file name / stored patient ID, or pass `--age`/`--sex`.
- "missing leads": the file lacks chest leads or two limb leads; nothing to derive them from.
- "segmentation failed: lead is flat or empty" / "no R peaks detected": the explained lead is
  zero-filled or has no detectable beats; explain another lead (`--xai_lead`).
- LIME is the slowest method (1000 model passes), SmoothGrad
  the next (50 passes): use `--xai gradcam` for large batches. `--xai all` costs ~1.9 s per
  record with figures (~0.5 s before LIME; the lead is delineated once for all methods) and
  writes four figures per record: add `--no-plots` if only the table is needed.
  `--lime_samples 500` halves LIME's time, but the map then changes visibly with the seed.
- "Could not load Cardiac-Ischemia-XNet": the message names the cause — torch < 2.6, or
  `models/*.pth` is a Git LFS pointer (run `git lfs pull`). `python3 ecg_xai.py` checks the
  packages first and says which are missing, broken or too old.