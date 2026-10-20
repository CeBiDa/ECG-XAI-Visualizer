[![GitHub ECG-XAI-Visualizer](https://img.shields.io/badge/github-ECG--XAI--Visualizer-blue?logo=github)](https://github.com/CeBiDa/ECG-XAI-Visualizer)
[![Python](https://img.shields.io/badge/Python-100_%25-blue?logo=python&logoColor=fff)](#) [![OpenCode](https://img.shields.io/badge/OpenCode-Skill-blue?logo=opencode&logoColor=fff)](#) [![ClaudeCode](https://img.shields.io/badge/Claude_Code-Skill-blue?logo=claudecode&logoColor=fff)](#)

# ECG-XAI-Visualizer

This Python reads raw 12-lead ECG data, applies a diagnostic CNN (in this case cardiac ischemia detection) and shows why the model decides for presenece or abscence of the diagnosis using explainable AI (**XAI**). Four XAI methods are implemented: **Grad-CAM**, **saliency maps**, **SmoothGrad×SIGN** and **LIME**. Based on those XAI models the parts of the ECG signal that drove the prediction are marked. The XAI based attributions are then split across the ECG segments *P wave*, *QRS complex*, *ST segment*, and *T wave*. So each prediction comes with a statement such as "40 % of the top attribution lies in the ST segment". 

```
ECG files (mat / wfdb / csv / asc / dcm / xml / hl7)  ──►  ischemia probability + XAI attribution per record  +  xai_summary.csv  +  <record>_xai[_<method>].png
```

ECG-XAI-Visualizer uses the same ECG readers and filters as our [ECG-Format-Converter](https://github.com/CeBiDa/ECG-Format-Converter) app [DOI: 10.5281/zenodo.22871601].

---

## Table of contents

- [Downloads](#downloads)
- [Features](#features)
- [Quick start](#quick-start)
  * [Desktop app (GUI)](#desktop-app-gui)
  * [Command line](#command-line)
  * [Python API](#python-api)
  * [AI agent integration](#ai-agent-integration)
- [Outputs](#outputs)
- [How it works](#how-it-works)
- [The model](#the-model)
- [Explainability](#explainability)
- [Technical stack](#technical-stack)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [License](#license)

---

## Downloads

Ready-to-use executables of the cross-platform GUI app are built by GitHub Actions and published in the [**Releases**](https://github.com/CeBiDa/ECG-XAI-Visualizer/releases) tab:

| Platform                    | Artifact                                 |
| --------------------------- | ---------------------------------------- |
| Windows x86_64              | `ECG-XAI-Visualizer-windows-x86_64.zip`  |
| macOS Apple Silicon (arm64) | `ECG-XAI-Visualizer-macos-arm64.app.zip` |
| Linux x86_64                | `ECG-XAI-Visualizer-linux-x86_64.zip`    |

Download, unzip, and run. No Python installation required. Builds are produced with PyInstaller and include the CPU build of PyTorch and the model weights.

## Features

- **Seven input formats**: MATLAB `.mat` (keys `template`/`val`/`ecg`), WFDB `.dat`/`.hea` pairs, CSV and ASC (comma, semicolon, tab or space separated), DICOM waveform `.dcm`, XML (`wavedata`/`ECG_RHYTHMS` schema with `patdata` demographics), and HL7 aECG (`AnnotatedECG`)
- **Ischemia prediction**: a residual 1-D CNN that reads 3 s of 12-lead ECG plus age and sex and returns P(ischemia); records at or above the decision threshold (0.564) are labelled *Ischemia*
- **Model input built the way the model was trained**: leads matched **by name** and put in the order I, II, III, aVR, aVL, aVF, V1–V6 (vendor names such as `Lead aVR, augmented voltage, right arm` or `MDC_ECG_LEAD_V1` are recognised), missing limb leads derived (Einthoven / Goldberger), the record resampled to 500 Hz, a 3 s window cut (start configurable) and each lead z-scored
- **Age and sex from the file first**: DICOM `PatientAge`, birth date + exam date (DICOM, HL7, XML), MAT `age`/`sex`, WFDB header comments (`Age: 74`, `Sex: Male`, or `<age>: 74`, `<sex>: M`); then an optional **metadata CSV** (ID, age, sex); then **fallback values** for the run. Where each value came from is recorded
- **Four XAI methods**: Grad-CAM, saliency map, SmoothGrad×SIGN and LIME, one of them or all four per run (the desktop app runs all four), always explaining the label the app reports; SmoothGrad and LIME are seeded, so a re-run gives the same map
- **Segment attribution**: NeuroKit2 delineates the lead into P / QRS / ST / T and the top-10 % attribution is split across the segments
- optional **Signal-processing pipeline**: IIR notch (50 or 60 Hz), Butterworth bandpass 0.5–40 Hz, wavelet baseline removal (db4), EMD denoising; steps run in the order given; optional **resampling** of the processed signal
- **Cross-platform desktop GUI** with live progress, per-file log, analysed/ischemia/skipped/failed counters, a record viewer (raw, processed, model input and XAI views; lead toggles; zoom and pan; a details panel) whose XAI view switches between the four methods instantly and explains any other lead on demand, and a sortable results table
- **Outputs**: `xai_summary.csv` (one row per analysed record, with the segment shares of every method side by side), one `<record>_xai.png` figure per record, optional `<record>_xai.npz` with the model input and the attribution map (`<record>_xai_<method>.png` / `.npz` with `--xai all` and in the desktop app)
- optional **Anonymization**: patient ID, names, birth date and exam date and time are left out of the results
- **Batch behavior**: each file fails or is skipped on its own, with the reason logged; one broken export never stops the run

## Quick start

### Desktop app (GUI)

1. Download the executable for your OS (Windows/macOS/Linux) from the [Releases](https://github.com/CeBiDa/ECG-XAI-Visualizer/releases) page (or run `python ui.py`).
2. Select an **input** folder or single file. The app scans it immediately and shows a per-format count. You can also drop a file or folder anywhere on the window.
3. Select an **output folder**; tick **Save XAI figures** for one PNG per record and XAI method.
4. Tick the **pre-processing** steps you want. Under **Sampling**, tick *Input sampling rate* for CSV/ASC files (they store none — the panel tells you how many of the selected files need it). Under **Age and sex**, add a **metadata CSV** or fallback values for files that carry no demographics. The explanation needs no settings: every record is explained with all four XAI methods, on the lead set in `config.ini` (II by default).
5. Hit **Start analysis**. Progress, per-file log lines (prediction and the segment split of each method) and counters update live; when the run ends the **Viewer** opens on the first record. Switch between *Raw*, *Processed*, *Model input* and *XAI*, toggle leads, scroll to zoom and drag to pan. In the *XAI* view, pick the **method** (all four are ready, so switching is instant) and the **lead** (another lead is explained on demand). The **Summary** table has one row per record; double-click a row to open the record in the viewer, or a cell in a method's columns (`gradcam_…`, `saliency_…`, `smoothgrad_…`, `lime_…`) to see that explanation. Cancel stops after the current file and keeps everything already written.

### Command line

```
git clone https://github.com/CeBiDa/ECG-XAI-Visualizer && cd ECG-XAI-Visualizer

# create & activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .venv\Scripts\activate             # Windows (cmd: .venv\Scripts\activate.bat)

pip install -r requirements.txt      # Python 3.10+ (Linux, Windows x64, macOS Apple Silicon)
# CPU-only PyTorch is enough and much smaller; install it first if you like:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
# PyTorch 2.6+ (needed for the safe weight loading) has no pip wheels for Intel Macs.

# analyse a folder: Grad-CAM on lead II, one figure per record
python3 ecg_xai.py data/input_ecgs --output data/results

# filter first, explain lead V5 with SmoothGrad x SIGN, 500 Hz for CSV/ASC files
python3 ecg_xai.py data/input_ecgs --output data/results --fs 500 \
    --pipeline notch bandpass --xai smoothgrad --xai_lead V5

# LIME with its maps: one signed weight per lead and 50 ms, ~1.2 s per record
python3 ecg_xai.py data/input_ecgs --output data/results --xai lime --save-maps

# all four XAI methods: one summary row per record, one figure per record and method
python3 ecg_xai.py data/input_ecgs --output data/results --xai all

# files without demographics: an ID/age/sex table, and fallbacks for the rest
python3 ecg_xai.py my_exports/ --output out/ --metadata patients.csv --age 60 --sex female

# predictions only, no figures, anonymized
python3 ecg_xai.py my_exports/ --output out/ --xai none --anonymize
```

Run `python3 ecg_xai.py` with no arguments for interactive prompts, and `--help` for the full flag list (`--threshold`, `--window_start`, `--allow-missing-leads`, `--save-maps`, `--lime_samples`, `--seed`, `--device`, `--config`, the EMD options, ...)

### Python API

```python
import sys
sys.path.insert(0, "/path/to/ECG-XAI-Visualizer")   # the repo root (not pip-installable)

from runner.execution_runner import ExecutionRunner

runner = ExecutionRunner(
    config_path="/path/to/ECG-XAI-Visualizer/config.ini",   # optional, defaults to the repo config; must exist if given
    path_source="path/to/input_folder",   # file or folder
    path_sink="path/to/output_folder",
)
runner.pipeline = ["notch", "bandpass"]   # any of notch, bandpass, wavelet, emd
runner.default_fs = 500                   # used when a file carries no rate
runner.metadata_csv = "patients.csv"      # ID / age / sex table (optional)
runner.fallback_age, runner.fallback_sex = 60, "Female"   # last resort (optional)
runner.xai_method = "gradcam"             # gradcam | saliency | smoothgrad | lime | all | none
runner.xai_lead = "II"
runner.lime_samples = 1000                # LIME: perturbed copies (model passes) per record
runner.keep_signals = True                # retain signals and maps in the returned dict

records = runner.run()
# records: {record_id: {"prediction": {"probability", "label", "threshold", ...},
#                       "xai": {(method, lead): {"map": (12, 1500), "segments": {...}}},
#                       "raw" / "processed": signals, "patient": age/sex + source, ...}}
# runner.summary      -> the rows of xai_summary.csv
# runner.stats        -> {"success": ..., "skipped": ..., "failed": ..., "positive": ...}
# runner.summary_path -> path of xai_summary.csv
```

`run()` also takes `progress_cb(done, total, record_id)`, `log_cb(text)`, and `cancel_cb() -> bool` for embedding in other tools; the GUI is built on exactly this interface. `ecgxai.analyze.explain_record(load_model(), records[rid], "saliency", "V5")` explains a kept record with another method or lead, reusing what the run already computed for it (each method's map, each lead's delineation).

### AI agent integration

The repository ships agent-ready instructions. [`CLAUDE.md`](CLAUDE.md) provides project context and commands for coding agents, and [`skills/ecg-xai-visualizer/SKILL.md`](skills/ecg-xai-visualizer/SKILL.md) is a portable [agent skill](https://code.claude.com/docs/en/skills) that lets an agent operate the tool autonomously: setup, analysis of arbitrary folders, result verification, and troubleshooting.

The skill is already installed for both supported harnesses, so cloning the repo is enough — no copying required:

| Harness     | Path                                                                           | Notes                                                          |
| ----------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| Claude Code | [`.claude/skills/ecg-xai-visualizer/`](.claude/skills/ecg-xai-visualizer/)     | Picked up automatically in this repo                           |
| opencode    | [`.opencode/skills/ecg-xai-visualizer/`](.opencode/skills/ecg-xai-visualizer/) | opencode also reads `.claude/skills/` as a documented fallback |

To make the skill available in *every* project, install it into your user-level skill directories (`~/.claude/skills/` and `~/.config/opencode/skills/`):

```bash
sh skills/sync.sh --user
```

Both in-repo copies are generated from the canonical file under `skills/`. After editing it, run `sh skills/sync.sh` to propagate the change, or `sh skills/sync.sh --check` (exit code 1 on drift) to verify them in CI.

## Outputs

When an XAI method is set, the tool writes one figure per analysed record, `<record>_xai.png` (unless `--no-plots`), and, with `--save-maps`, `<record>_xai.npz` with the arrays behind it. With `--xai all`, and always in the desktop app, each method gets its own files: `<record>_xai_gradcam.png`, `<record>_xai_saliency.png`, `<record>_xai_smoothgrad.png` and `<record>_xai_lime.png` (and `.npz`). Records in subfolders are named `<subfolder>__<record>`; two inputs with the same name in one folder (`same.csv`, `same.mat`) get the extension added (`same_csv_xai.png`). Names longer than 180 bytes keep their end plus a short hash; the `<method>_plot` columns always give the exact names of a record's figures.

- **Figure**: the explained lead's 3 s model input with the P / QRS / ST / T segments shaded and the top-10 % attributions marked, the attribution curve underneath (with the zero line of a signed map; in 50 ms steps for LIME), and the share of the top attribution per segment as bars; the title carries the prediction, the probability, the threshold and the class explained.
- **Maps (`.npz`)**: `model_input` (12 × 1500, 500 Hz, leads I..V6, before the per-lead z-score the model applies; fewer samples for a record shorter than 3 s), `attribution` (same shape, scaled to [0, 1]; Grad-CAM repeats its single time map on every lead, LIME holds one value per lead and 50 ms), `segment_labels` (1500; 0 = P, 1 = QRS, 2 = ST, 3 = T, −1 = other) of the explained `lead`, `top_percentile`, `zero_level` (where no attribution lies in the map: 0, or the zero of a signed map: about 0.5 for SmoothGrad×SIGN, wherever a weight of 0 lands for LIME), `xai_note`, `leads`, `fs`, `window_start_s`, `probability`, `threshold`, `method`, `target`.

A consolidated **`xai_summary.csv`** is written to the output folder, one row per analysed record however many methods run, with these 50 columns:

| Column                                                                                        | Content                                                                                                                                                                                                                                              |
| --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `filename`                                                                                    | record id: path relative to the input folder (WFDB pairs under their `.hea` name)                                                                                                                                                                    |
| `prediction`, `probability`, `threshold`                                                      | label, P(ischemia) (6 decimals; the label is decided on the unrounded value), decision threshold                                                                                                                                                     |
| `model`                                                                                       | model name                                                                                                                                                                                                                                           |
| `xai_methods`, `xai_lead`, `xai_target`                                                       | the methods requested (comma-separated: `gradcam,saliency,smoothgrad,lime` with `all`), the lead explained, and the class explained (= `prediction` whenever an explanation ran)                                                                     |
| `<method>_p_wave`, `<method>_qrs`, `<method>_st_segment`, `<method>_t_wave`, `<method>_other` | % of the method's top-10 % attribution in each segment: five columns each for `gradcam`, `saliency`, `smoothgrad` and `lime`, in that order (blank for a method that did not run)                                                                    |
| `delineation`                                                                                 | NeuroKit2 delineator used (`dwt`, `prominence`) or `window` (fixed fallback windows; `xai_note` says why); all methods share the lead's delineation                                                                                                  |
| `xai_note`                                                                                    | the lead's delineation note (it could not be segmented, fell back to fixed windows, or bounded the ST segment in too few beats, so the ST share is a lower bound), and why a method gave no segment split or failed (after the method's name), if so |
| `patient_id`                                                                                  | from the file (blank with `--anonymize`)                                                                                                                                                                                                             |
| `patient_age`, `age_source`, `patient_sex`, `sex_source`                                      | what the model was given, and from where: `file`, `csv`, `fallback`                                                                                                                                                                                  |
| `exam_date`, `exam_time`                                                                      | from the file (blank with `--anonymize`)                                                                                                                                                                                                             |
| `sampling_freq`, `fs_source`, `n_samples`                                                     | the file's rate (`file`) or the asserted one (`fallback`), and its length                                                                                                                                                                            |
| `original_leads`, `calculated_leads`, `missing_leads`                                         | leads recorded, derived, zero-filled                                                                                                                                                                                                                 |
| `pipeline`                                                                                    | pre-processing steps (and resampling) applied                                                                                                                                                                                                        |
| `window_start_s`                                                                              | where the 3 s model window starts                                                                                                                                                                                                                    |
| `notes`                                                                                       | anything unusual: window moved, short record, ignored channels, zero-filled leads, a notch skipped at a low sampling rate, leads cut short by a truncated export                                                                                     |
| `gradcam_plot`, `saliency_plot`, `smoothgrad_plot`, `lime_plot`                               | the figure written for each method (blank if none)                                                                                                                                                                                                   |

A method's five shares add up to 100 % (± rounding). They are blank, and `xai_note` says why, when the lead could not be segmented (`delineation` is blank too) or when the method put no positive attribution in the lead's top 10 % (`LIME: no positive attribution in the top 10 % of this lead`). A method that fails leaves its columns blank and adds `<method name>: explanation failed: <error>` to `xai_note`; the other methods' results are kept. Skipped and failed files get no row; the run log says why. Note that ID and exam date are **personal health information**; handle outputs accordingly or run with `--anonymize`.

## How it works

```
                     ┌──────────────────────────────────────────────────┐
ECG file ──► reader  │ mat | wfdb | csv | asc | dcm | xml | hl7         │
                     │ waveforms (leads x samples) + metadata dict      │
                     └───────────────────┬──────────────────────────────┘
                                         ▼
     ┌───────────────────────────────────────────────────────────────────┐
     │ 1. sampling rate: the file's, else --fs (none: skipped)           │
     │ 2. age & sex: file -> metadata CSV -> fallback (none: skipped)    │
     │ 3. keep the standard leads (matched by name), fill NaN gaps       │
     │ 4. pipeline (optional, in order): notch -> bandpass ->            │
     │    wavelet baseline removal -> EMD denoise; resample (optional)   │
     │ 5. derive missing leads; order I..V6; leads still missing:        │
     │    skip the record (or zero-fill with --allow-missing-leads)      │
     │ 6. model input: record at 500 Hz, 3 s window, z-score per lead    │
     └───────────────────────────────────┬───────────────────────────────┘
                                         ▼
     ┌───────────────────────────────────────────────────────────────────┐
     │ CNN Model (ECG + age + sex) -> P(ischemia) -> label   │
     │ XAI for that label: Grad-CAM | saliency | SmoothGrad x SIGN       │
     │                     | LIME | all                                  │
     │ NeuroKit2 delineation -> share of the top 10 % attribution in     │
     │ P wave, QRS, ST segment, T wave                                   │
     └───────────────────────────────────┬───────────────────────────────┘
                                         ▼
                 xai_summary.csv row  +  <record>_xai.png  (+ .npz)
                   (all: one <record>_xai_<method>.png per method)
```

## The model

The **CNN Model** used as an example classifies ischemia vs. no ischemia. A dilated multi-scale convolutional stem (kernels 3, 5, 7) feeds three bottleneck residual blocks with squeeze-and-excitation and stochastic depth (128 → 256 channels, each halving the time axis); after global average pooling, age and sex are appended and a dropout + linear layer returns two logits. Activations are ReLU. It has 207,713 parameters. The weights ship in `models/finetuned_ischemia_robust_model.pth`, the robust fine-tuned release: a stochastic weight average (SWA) of 20 snapshots, saved as the state_dict of PyTorch's `AveragedModel`.

| Input  | Value                                                                                                                                                                                                                    |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ECG    | 12 leads in the order I, II, III, aVR, aVL, aVF, V1–V6; 500 Hz; 3 s (1500 samples); each lead z-scored                                                                                                                   |
| Age    | z-scored with the training mean 56.47 and SD 18.35 years                                                                                                                                                                 |
| Sex    | Male = 1, Female = 0                                                                                                                                                                                                     |
| Output | softmax; P(ischemia) at or above **0.564** is labelled *Ischemia* (the deployed threshold of the original app, chosen on its test set by Youden's J / F1; the checkpoint stores none of its own). `threshold` changes it |

The checkpoint is read with `torch.load(..., weights_only=True)`: nothing in a weights file can execute code. The model catalogue in `ecgxai/registry.py` describes every input requirement, so further models can be added there.

## Explainability

Each method returns a map over the 12 × 1500 input, scaled to [0, 1] per record, and always explains the class the app reports (the label at the decision threshold).

| Method              | What it shows                                                                                                                                                                                                                                                                                                                                                                  | Implementation based on                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| **Grad-CAM**        | Gradient-weighted activations of the last convolutional stage (`layer3`), interpolated back to 1500 samples. One map over **time**, shared by all leads: the stage has no lead axis left, and its time resolution is about 16 ms                                                                                                                                               | Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2016). <br>Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. <br>arXiv preprint arXiv:1610.02391                                                                                                                                                                          |
| **Saliency map**    | Absolute gradient of the score with respect to every input sample: a map per **lead and sample**                                                                                                                                                                                                                                                                               | Simonyan, K., Vedaldi, A., & Zisserman, A. (2013).<br>Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.<br>arXiv preprint arXiv:1312.6034.                                                                                                                                                                                                   |
| **SmoothGrad×SIGN** | The gradient averaged over 50 copies of the input with Gaussian noise (SD 0.2 in z-score units), times the sign of the input. Steadier than a plain saliency map; seeded (`seed = 0`), so the same record always gives the same map                                                                                                                                            | Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017).<br>SmoothGrad: Removing noise by adding noise.<br> arXiv preprint arXiv:1706.03825. <br><br/>Gumpfer, N., Prim, J., Keller, T., Seeger, B., Guckert, M., & Hannig, J. (2023).<br>SIGNed explanations: Unveiling relevant features by reducing bias<br>Information Fusion, 99, 101883.                              |
| **LIME**            | A local surrogate model over 50 ms pieces of each lead: the model scores 1000 copies of the record, each with about 1 % of the pieces set to the lead's median, and a weighted ridge regression of the log-odds on which pieces were kept gives one signed weight per **lead and 50 ms**. It explains the recorded signal, including the per-lead z-score; seeded (`seed = 0`) | Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).<br> "Why Should I Trust You?": Explaining the Predictions of Any Classifier<br> arXiv: 1602.04938. KDD '16. <br/><br>Bodini, M., Rivolta, M. W., & Sassi, R. (2020).<br>Interpretability Analysis of Machine Learning Algorithms in the Detection of ST-Elevation Myocardial Infarction<br>2020 Computing in Cardiology (CinC), pp. 1–4 |

**Segment attribution.** The explained lead is delineated with NeuroKit2 (`ecg_clean`, `ecg_peaks`, then `ecg_delineate` with the methods `dwt` and `prominence` in turn, on the lead as recorded and, when its QRS points down as in aVR or V1, also inverted, since NeuroKit2 expects an upright QRS; the first result that bounds the QRS and the T wave of at least half the beats wins) into P wave, QRS complex (R onset to R offset), ST segment (R offset to T onset) and T wave. The samples at or above the 90th percentile of the lead's attribution — the *top 10 %* — are kept, and each segment's share of their summed attribution is reported (for the signed maps, SmoothGrad×SIGN and LIME, the part above their zero level: about 0.5 of the scaled map for SmoothGrad×SIGN, wherever a weight of 0 lands for LIME); whatever falls outside the segments is *Other*. Where segments would overlap, a sample counts once (QRS > ST > T > P), so the shares add up to 100 %.

Delineation runs on the **whole record** at 500 Hz and is then cut to the 3 s window: NeuroKit2's `dwt` delineator refuses signals shorter than 4 s. Records under 4 s are therefore usually delineated by `prominence` (NeuroKit2 0.2.11 or later). Only when every delineator fails or leaves the QRS or T wave of most beats unbounded (as `dwt` does above about 160 bpm) are fixed windows around each R peak used (P −200 to −60 ms, QRS ±50 ms, ST +50 to +150 ms, T +150 to +400 ms); `delineation` then says `window` and `xai_note` why.

## Technical stack

| Layer             | Technology                                                                         |
| ----------------- | ---------------------------------------------------------------------------------- |
| Language          | Python 3.10+                                                                       |
| Model & XAI       | PyTorch (Grad-CAM, saliency, SmoothGrad×SIGN, LIME implemented directly)           |
| ECG delineation   | NeuroKit2                                                                          |
| Signal processing | NumPy, SciPy (`iirnotch`, `butter`/`filtfilt`, `resample`), PyWavelets, EMD-signal |
| Format I/O        | pydicom (DICOM), wfdb (WFDB), XML, SciPy (MAT), pandas (CSV/ASC)                   |
| Desktop GUI       | PySide6 (Qt6), dark theme, threaded workers                                        |
| Visualization     | Matplotlib (viewer on a QtAgg canvas, figures via Agg)                             |
| Progress UX       | tqdm (CLI), Qt signal bridge (GUI)                                                 |
| Packaging         | PyInstaller (`.spec` included) + GitHub Actions multi-platform builds              |

Dependencies are listed in [`requirements.txt`](requirements.txt).

## Configuration

`config.ini` sets the defaults for the CLI and the GUI. Command-line flags and GUI controls override it per run.

Section `[processing]`:

| Setting           | Type   | Default | Description                                                                                                  |
| ----------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------------ |
| `pipeline`        | list   | empty   | Comma-separated steps: `notch`, `bandpass`, `wavelet`, `emd`.                                                |
| `notch`           | int    | `50`    | Notch filter frequency, 50 or 60 Hz.                                                                         |
| `default_fs`      | float  | blank   | Sampling rate for files that carry none; blank = such files are skipped.                                     |
| `resample`        | float  | blank   | Resample the processed signal; blank keeps the file's rate. The model input is brought to 500 Hz regardless. |
| `resample_method` | string | `fft`   | `fft` or `interpolation`; also used for the model input.                                                     |
| `derive_leads`    | bool   | `True`  | Derive missing leads (I, II, III, aVR, aVL, aVF) from the recorded ones.                                     |
| `anonymize`       | bool   | `False` | Leave patient ID, names, birth date and exam date/time out of the results.                                   |

Section `[model]`:

| Setting               | Type   | Default                 | Description                                                                                                   |
| --------------------- | ------ | ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| `name`                | string | `cardiac-ischemia-xnet` | Model to run.                                                                                                 |
| `threshold`           | float  | `0.564`                 | Decision threshold on P(ischemia); blank = the model's default.                                               |
| `window_start`        | float  | `0`                     | Seconds into the record where the 3 s window starts.                                                          |
| `allow_missing_leads` | bool   | `False`                 | Run records with leads that are neither recorded nor derivable on zero-filled leads instead of skipping them. |
| `device`              | string | `cpu`                   | `cpu`, `cuda`, `mps` or `auto`.                                                                               |

Section `[patient]` — age and sex come from the file first, then these:

| Setting        | Type   | Default | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `metadata_csv` | path   | blank   | CSV (comma or semicolon) with an ID column and `age` and/or `sex` columns (`basis_age`/`basis_sex`, `gender` also accepted). A plain `ID` column matches a record by the id the run reports (`sub/rec.csv`, WFDB pairs as `rec.hea`) or its file name (with or without extension), the patient ID stored in the file, or the leading number of the file name. Record-level columns (`record_id`, `ecg_id`, `filename`, `study_id`) match only the record id, the file name or its leading number; patient-level ones (`patient_id`, `subject_id`) only the patient ID stored in the file. A unit in brackets in a header (`Age (years)`) is ignored; UTF-8 and Excel's Windows encoding are both read. |
| `fallback_age` | float  | blank   | Age in years for records with none from the file or CSV.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `fallback_sex` | string | blank   | `Male` or `Female` (`M`/`F`) for records with none.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

Section `[xai]`:

| Setting              | Type   | Default   | Description                                                                                                                                                                                                                                                                                  |
| -------------------- | ------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `method`             | string | `gradcam` | `gradcam`, `saliency`, `smoothgrad`, `lime`, `all` (every method: one figure per record and method, their segment shares side by side in the summary), or `none` for predictions only. The desktop app always computes all methods; this setting is for the command line and the Python API. |
| `lead`               | string | `II`      | Lead to explain and segment for the summary and the figures, with every method (the desktop app uses it too; its viewer explains any lead).                                                                                                                                                  |
| `top_percentile`     | float  | `90`      | Attributions at or above this percentile count as "top".                                                                                                                                                                                                                                     |
| `save_plots`         | bool   | `True`    | Write `<record>_xai.png` (with `all`: `<record>_xai_<method>.png`).                                                                                                                                                                                                                          |
| `save_maps`          | bool   | `False`   | Write `<record>_xai.npz` (with `all`: `<record>_xai_<method>.npz`).                                                                                                                                                                                                                          |
| `smoothgrad_samples` | int    | `50`      | Noisy copies for SmoothGrad×SIGN.                                                                                                                                                                                                                                                            |
| `smoothgrad_noise`   | float  | `0.2`     | Their SD, in z-score units.                                                                                                                                                                                                                                                                  |
| `lime_samples`       | int    | `1000`    | Perturbed copies of the record for LIME, one model pass each (~1.2 s per record).                                                                                                                                                                                                            |
| `seed`               | int    | `0`       | Seed for the SmoothGrad noise and the LIME perturbations; blank = new draws every run.                                                                                                                                                                                                       |

Section `[emd]` tunes the EMD step exactly as in ECG-Format-Converter (`method`, `remove_baseline`, `baseline_cut`, `noise_cut`, `threshold`, `thresholding`, `protect_qrs`, `qrs_halfwidth`, `qrs_threshold_scale`, `adaptive_window`, `trials`); its values are always checked (a bad one stops every run) but only used when `emd` is in the pipeline. The CLI exposes `--emd_method`, `--emd_threshold`, `--emd_noise_cut`, `--emd_soft`, `--emd_keep_baseline` and `--emd_no_qrs_protection`.

## Project structure

```
ECG-XAI-Visualizer/
├── config.ini                       # Run defaults
├── ecg_xai.py                       # CLI entry point
├── ui.py                            # PySide6 desktop app
├── debug_ecg.py                     # Input inspector (readers only, no model, writes nothing)
├── requirements.txt
├── ECG-XAI-Visualizer.spec          # PyInstaller build recipe
├── .github/workflows/build.yml      # Release binary CI (Win / macOS / Linux)
├── models/
│   └── finetuned_ischemia_robust_model.pth # CNN Model weights (robust fine-tuned, SWA)
├── runner/
│   └── execution_runner.py          # Orchestrator: discover -> analyse -> summarize
├── ecgxai/
│   ├── cli.py                       # Argument parsing + interactive prompts
│   ├── analyze.py                   # Per-file read -> pre-process -> predict -> explain
│   ├── registry.py                  # Model catalogue and safe weight loading
│   ├── architectures.py             # Cardiac-Ischemia-XNet network
│   ├── inference.py                 # Model input (500 Hz, 3 s, z-score) and prediction
│   ├── explain.py                   # Grad-CAM, saliency, SmoothGrad x SIGN, LIME
│   ├── segments.py                  # NeuroKit2 delineation, attribution per segment
│   ├── plotting.py, theme.py        # Figures shared by the viewer and the exports
├── ecgproc/
│   ├── readers.py                   # mat / wfdb / csv / asc / dcm / xml / hl7
│   ├── preprocess.py                # notch, bandpass, wavelet, EMD, resampling
│   ├── leads.py                     # Einthoven / Goldberger lead derivation
│   ├── standardize.py               # Lead names -> fixed 12-lead layout
│   ├── patient.py                   # Age and sex: file, metadata CSV, fallback
│   ├── files.py, constants.py, xml_utils.py, deps.py
├── skills/
│   ├── ecg-xai-visualizer/SKILL.md  # Canonical agent skill
│   └── sync.sh                      # Copies it into .claude/ and .opencode/
├── .claude/skills/                  # Claude Code discovery path (generated)
├── .opencode/skills/                # opencode discovery path (generated)
└── data/
    ├── input_ecgs/                  # <- put your ECG files here
    └── results/                     # <- figures and xai_summary.csv land here
```

## License

Released under the MIT License, see [LICENSE.md](LICENSE.md). 
