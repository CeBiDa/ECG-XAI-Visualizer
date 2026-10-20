import argparse
import os
import re
import sys

from tqdm import tqdm

from ecgproc.constants import lead_label
from ecgproc.deps import check_dependencies
from ecgproc.preprocess import EMD_DEFAULTS, PIPELINE_STEPS
from ecgxai.explain import XAI_METHODS
from ecgxai.registry import MODELS, get_spec
from runner.execution_runner import SUMMARY_NAME, XAI_CHOICES, ExecutionRunner, default_config_path


def _interactive():
    return sys.stdin.isatty()


def _opt(*names):
    out = []
    for n in names:
        out.append(n)
        alt = n.replace("_", "-") if "_" in n else None
        if alt and alt not in out:
            out.append(alt)
    return out


def _answer(prompt):
    answer = input(prompt).strip().strip('"\'').strip()
    if os.name != "nt" and "\\" in answer and not os.path.exists(answer):
        unescaped = re.sub(r"\\(.)", r"\1", answer)
        if os.path.exists(unescaped):
            return unescaped
    return answer


def _seed(value):
    return None if str(value).strip().lower() in ("none", "random", "") else int(value)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ecg_xai",
        description="Ischemia prediction and explainability (XAI) for 12-lead ECG recordings.",
    )
    parser.add_argument("input", nargs="?", help="Input ECG file or folder")
    parser.add_argument("--output", help="Output folder for xai_summary.csv and the figures")

    pre = parser.add_argument_group("Pre-processing")
    pre.add_argument("--pipeline", nargs="+", choices=list(PIPELINE_STEPS), default=None,
                     help="Filter steps, applied in the order given.")
    pre.add_argument("--notch", type=int, choices=[50, 60], default=None,
                     help="Notch filter frequency (50 or 60 Hz). Default 50, or config.ini.")
    pre.add_argument("--fs", type=float,
                     help="Sampling rate in Hz to assume for files that do not store one "
                          "(CSV, ASC, some MAT, XML/HL7 without a rate). Files that carry their "
                          "own rate keep it. Without it those files are skipped: the model needs "
                          "a rate.")
    pre.add_argument("--resample", type=float,
                     help="Resample the processed signal to this rate in Hz. The model input "
                          "is brought to the model's own rate (500 Hz) in any case.")
    pre.add_argument(*_opt("--resample_method"), dest="resample_method",
                     choices=["fft", "interpolation"], default=None,
                     help="Resampling method, also for the model input. Default fft.")
    pre.add_argument("--no-derive", dest="derive_leads", action="store_false", default=None,
                     help="Do not derive missing leads from the recorded ones.")

    emd = parser.add_argument_group("EMD denoising (only with --pipeline ... emd)")
    emd.add_argument(*_opt("--emd_method"), dest="emd_method", choices=["emd", "eemd", "ceemdan"],
                     default=None, help=f"Decomposition. Default '{EMD_DEFAULTS['method']}'.")
    emd.add_argument(*_opt("--emd_threshold"), dest="emd_threshold", type=float, default=None,
                     help=f"Noise threshold scale. Default {EMD_DEFAULTS['threshold']}.")
    emd.add_argument(*_opt("--emd_noise_cut"), dest="emd_noise_cut", type=float, default=None,
                     help=f"Hz above which an IMF counts as noise. Default {EMD_DEFAULTS['noise_cut']}.")
    emd.add_argument("--emd_soft", dest="emd_thresholding", action="store_const", const="soft",
                     default=None, help="Shrink thresholded intervals instead of keeping them intact.")
    emd.add_argument("--emd_keep_baseline", dest="emd_remove_baseline", action="store_false",
                     default=None, help="Keep the low-frequency trend.")
    emd.add_argument("--emd_no_qrs_protection", dest="emd_protect_qrs", action="store_false",
                     default=None, help="Threshold inside QRS complexes too (not recommended).")

    mdl = parser.add_argument_group("Model")
    mdl.add_argument("--model", choices=list(MODELS), default=None,
                     help="Model to run. Default from config.ini.")
    mdl.add_argument("--threshold", type=float, default=None,
                     help="Decision threshold on P(ischemia). Default from config.ini "
                          f"(blank there = the model's own, {get_spec().threshold:g}).")
    mdl.add_argument(*_opt("--window_start"), dest="window_start", type=float, default=None,
                     help="Seconds into the record where the 3 s model window starts. Default 0.")
    mdl.add_argument("--allow-missing-leads", dest="allow_missing_leads", action="store_true",
                     default=None, help="Run records whose missing leads cannot be derived on "
                                        "zero-filled leads instead of skipping them.")
    mdl.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default=None,
                     help="Where the model runs. Default cpu.")

    pat = parser.add_argument_group("Patient (age and sex: file first, then --metadata, then --age/--sex)")
    pat.add_argument("--metadata", help="CSV with an ID column plus age and/or sex columns.")
    pat.add_argument("--age", type=float, help="Age in years for records with none in file or CSV.")
    pat.add_argument("--sex", help="Male or Female (M/F) for records with none in file or CSV.")

    xai = parser.add_argument_group("Explainability")
    xai.add_argument("--xai", choices=list(XAI_CHOICES), default=None,
                     help="Attribution method; all = every method (one figure per method, their "
                          "segment shares side by side in the summary); none = predictions only. "
                          "Default gradcam.")
    xai.add_argument(*_opt("--xai_lead"), dest="xai_lead", default=None,
                     help="Lead to explain and segment. Default II.")
    xai.add_argument(*_opt("--top_percentile"), dest="top_percentile", type=float, default=None,
                     help="Percentile above which attributions count as top. Default 90.")
    xai.add_argument(*_opt("--smoothgrad_samples"), dest="smoothgrad_samples", type=int, default=None,
                     help="Noisy copies for SmoothGrad x SIGN. Default 50.")
    xai.add_argument(*_opt("--smoothgrad_noise"), dest="smoothgrad_noise", type=float, default=None,
                     help="Noise SD for SmoothGrad x SIGN (z-score units). Default 0.2.")
    xai.add_argument(*_opt("--lime_samples"), dest="lime_samples", type=int, default=None,
                     help="Perturbed copies of the record for LIME, one model pass each. "
                          "Default 1000; fewer give noisier maps.")
    xai.add_argument("--seed", type=_seed, default=argparse.SUPPRESS,
                     help="Seed for the SmoothGrad noise and the LIME perturbations, or 'none' "
                          "for new draws every run. Default 0.")
    xai.add_argument("--no-plots", dest="save_plots", action="store_false", default=None,
                     help="Do not write <record>_xai.png figures (<record>_xai_<method>.png with "
                          "--xai all).")
    xai.add_argument("--save-maps", dest="save_maps", action="store_true", default=None,
                     help="Also write <record>_xai.npz with the model input and attribution map "
                          "(<record>_xai_<method>.npz with --xai all).")

    parser.add_argument("--anonymize", action="store_true",
                        help="Leave patient ID, names, birth date and exam date/time out of the "
                             "results.")
    parser.add_argument("--config", help=f"Config file (default: {default_config_path()})")
    return parser


def main(argv=None):
    check_dependencies()
    args = build_parser().parse_args(argv)

    if not args.input:
        if not _interactive():
            print("Error: no input path given. Usage: ecg_xai.py <input> --output <folder>")
            return 2
        args.input = _answer("Enter input file or folder path: ")
        if not args.input:
            print("Error: no input path given.")
            return 2
    if not args.output and _interactive():
        args.output = _answer("Enter output folder (empty = xai_results): ")

    try:
        runner = ExecutionRunner(config_path=args.config, path_source=args.input,
                                 path_sink=args.output or "xai_results")
    except ValueError as e:
        print(f"Error: {e}")
        return 2

    simple = {
        "pipeline": "pipeline", "notch": "notch", "fs": "default_fs", "resample": "resample",
        "resample_method": "resample_method", "derive_leads": "derive_leads", "model": "model",
        "threshold": "threshold", "window_start": "window_start",
        "allow_missing_leads": "allow_missing_leads", "device": "device",
        "metadata": "metadata_csv", "age": "fallback_age", "sex": "fallback_sex",
        "xai": "xai_method", "xai_lead": "xai_lead", "top_percentile": "top_percentile",
        "smoothgrad_samples": "smoothgrad_samples", "smoothgrad_noise": "smoothgrad_noise",
        "lime_samples": "lime_samples", "save_plots": "save_plots", "save_maps": "save_maps",
    }
    for arg, attr in simple.items():
        value = getattr(args, arg)
        if value is not None:
            setattr(runner, attr, value)
    if "seed" in vars(args):
        runner.seed = args.seed
    if args.anonymize:
        runner.anonymize = True
    for key in ("method", "threshold", "noise_cut", "thresholding", "remove_baseline", "protect_qrs"):
        value = getattr(args, f"emd_{key}")
        if value is not None:
            runner.emd_options[key] = value

    files = runner.input_files()
    if not files:
        print(f"No ECG files found in {runner.path_source}")
        return 1
    try:
        opts = runner.options()
    except ValueError as e:
        print(f"Error: {e}")
        return 2

    methods = opts.xai_methods
    print(f"\nFound {len(files)} files to analyse.")
    print(f"Output directory: {runner.path_sink}")
    print(f"Model: {runner.model}  (threshold {opts.threshold:g}, "
          f"window from {opts.window_start:g} s)")
    if runner.pipeline:
        print(f"Pre-processing: {' -> '.join(runner.pipeline)}"
              + (f" (notch {runner.notch} Hz)" if "notch" in runner.pipeline else ""))
    else:
        print("Pre-processing: none")
    if runner.default_fs:
        print(f"Assumed sampling rate for files without one: {runner.default_fs:g} Hz")
    if runner.resample:
        print(f"Processed signal resampled to {runner.resample:g} Hz ({runner.resample_method})")
    print(f"Derive missing leads: {'Yes' if runner.derive_leads else 'No'}")
    sources = ["file"] + (["metadata CSV"] if runner.metadata_csv else [])
    if opts.fallback_age is not None or opts.fallback_sex is not None:
        age = f"{opts.fallback_age:g}" if opts.fallback_age is not None else "-"
        sources.append(f"fallback (age {age}, sex {opts.fallback_sex or '-'})")
    print(f"Age and sex from: {' -> '.join(sources)}")
    if methods:
        outs = [o for o, on in (("figures", opts.save_plots), ("maps", opts.save_maps)) if on]
        names = ", ".join(XAI_METHODS[m] for m in methods)
        print(f"Explanation: {names} on lead {lead_label(opts.xai_lead)}"
              + (f" ({', '.join(outs)})" if outs else " (summary only)"))
    else:
        print("Explanation: none (predictions only)")
    print(f"Anonymize: {'Yes' if runner.anonymize else 'No'}")
    print("-" * 30)

    bar = tqdm(total=len(files), desc="Analysing ECG files", unit="file")

    def _progress(done, total, name):
        bar.n = done
        bar.refresh()

    try:
        runner.run(progress_cb=_progress, log_cb=tqdm.write)
    except (ValueError, OSError, RuntimeError) as e:
        bar.close()
        print(f"\nError: {e}")
        return 2
    finally:
        bar.close()

    print("\n### Summary ###")
    print(f"Analysed: {runner.stats['success']} ({runner.stats['positive']} predicted ischemia)")
    print(f"Skipped: {runner.stats['skipped']}")
    print(f"Failed: {runner.stats['failed']}")
    if runner.summary_error:
        print(f"Error: {runner.summary_error}")
    if runner.summary_path:
        print(f"Results: {runner.summary_path}")
    elif not runner.summary_error:
        print(f"No record was analysed, so no {SUMMARY_NAME} was written.")

    if runner.summary_error:
        return 1
    return 1 if runner.stats['failed'] and not runner.stats['success'] else 0