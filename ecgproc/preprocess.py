"""
Signal pre-processing for ECG records.

Every filter accepts a single lead (1-D) or a (n_leads, n_samples) array and
returns the same shape and dtype family, so the functions can be used both
inside :func:`preprocess_signals` and stand-alone.
"""
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, resample, sosfiltfilt

try:
    from PyEMD import EMD
    _PYEMD_AVAILABLE = True
except ImportError:
    _PYEMD_AVAILABLE = False


# --------------------------------------------------------------------- shapes

def _as_2d(signal):
    arr = np.asarray(signal)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :], True
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1-D lead or a 2-D (n_leads, n_samples) array, got shape {arr.shape}.")
    return arr, False


def _restore(out, was_1d):
    return out[0] if was_1d else out


def _filtfilt(b, a, arr, fs=None):
    n = arr.shape[1]
    default_pad = 3 * max(len(a), len(b))
    if fs:
        default_pad = max(default_pad, int(round(3.0 * float(fs))))
    if n <= 3:
        return arr.copy()
    return filtfilt(b, a, arr, axis=1, padlen=min(default_pad, n - 1))


# ------------------------------------------------------------ linear filters

def wavelet_baseline_removal(ecg_signal, wavelet='db4', level=8, fs=None):
    arr, was_1d = _as_2d(ecg_signal)
    n = arr.shape[1]
    if fs:
        level = level + int(round(np.log2(float(fs) / 500.0)))
    usable = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    level = max(1, min(level, usable)) if usable >= 1 else 0
    if level == 0:
        return _restore(arr.copy(), was_1d)

    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        coeffs = pywt.wavedec(arr[i, :], wavelet, level=level)
        baseline = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]],
                                wavelet)[:n]
        out[i, :] = arr[i, :] - baseline
    return _restore(out, was_1d)


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=500, order=3):
    arr, was_1d = _as_2d(signal)
    nyq = 0.5 * float(fs)
    if nyq <= 0:
        raise ValueError("Sampling frequency must be positive.")

    low = max(float(lowcut), 0.0) / nyq
    high = float(highcut) / nyq
    if low >= 1.0:
        raise ValueError(f"Bandpass low cut-off {lowcut} Hz is at or above Nyquist ({nyq} Hz).")

    if high >= 1.0:
        b, a = butter(order, low, btype='highpass', analog=False) if low > 0 else (None, None)
        if b is None:
            return _restore(arr.copy(), was_1d)
    else:
        b, a = butter(order, [low, high], btype='band', analog=False)
    return _restore(_filtfilt(b, a, arr, fs), was_1d)


def notch_filter(signal, notch_freq, fs=500, quality_factor=30):
    arr, was_1d = _as_2d(signal)
    nyq = 0.5 * float(fs)
    norm_notch = float(notch_freq) / nyq
    if not 0 < norm_notch < 1:
        raise ValueError(f"Notch frequency {notch_freq} Hz is outside (0, {nyq}) Hz for fs={fs} Hz.")
    b, a = iirnotch(norm_notch, quality_factor)
    return _restore(_filtfilt(b, a, arr, fs), was_1d)


# EMD denoising
EMD_DEFAULTS = {
    "method": "emd",           # emd | eemd | ceemdan
    "remove_baseline": True,   # drop sub-baseline_cut IMFs and the trend
    "baseline_cut": 0.5,       # Hz — below this an IMF is baseline wander
    "noise_cut": 20.0,         # Hz — above this an IMF is noise-dominated
    "threshold": 1.0,          # universal-threshold scale (0 = keep noise IMFs whole)
    "thresholding": "hard",    # hard | soft interval thresholding
    "protect_qrs": True,       # relax the threshold inside every QRS complex
    "qrs_halfwidth": 0.06,     # s — protected half-window around each R peak
    "qrs_threshold_scale": 0.5,  # threshold multiplier inside that window (0 = keep everything)
    "adaptive_window": 0.5,    # s — window the noise level is tracked over (0 = one global estimate)
    "max_imf": None,           # cap on the number of IMFs (None = decompose fully)
    "trials": 30,              # ensemble size for eemd / ceemdan
    "noise_width": 0.05,       # ensemble noise amplitude, relative to signal range
    "seed": 0,                 # ensemble noise seed (reproducible runs)
}

_MAD_TO_SIGMA = 1.0 / 0.6745


def merge_emd_options(options):
    opt = dict(EMD_DEFAULTS)
    if options:
        unknown = set(options) - set(EMD_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown EMD option(s): {', '.join(sorted(unknown))}")
        opt.update({k: v for k, v in options.items() if v is not None or k in ("seed", "max_imf")})
    opt["method"] = str(opt["method"]).strip().lower()
    if opt["method"] not in ("emd", "eemd", "ceemdan"):
        raise ValueError(f"Unknown EMD method '{opt['method']}'. Choose from 'emd', 'eemd', 'ceemdan'.")
    opt["thresholding"] = str(opt["thresholding"]).strip().lower()
    if opt["thresholding"] not in ("hard", "soft"):
        raise ValueError(f"Unknown thresholding '{opt['thresholding']}'. Choose from 'hard', 'soft'.")
    opt["threshold"] = max(0.0, float(opt["threshold"]))
    opt["baseline_cut"] = max(0.0, float(opt["baseline_cut"]))
    opt["noise_cut"] = max(0.0, float(opt["noise_cut"]))
    opt["qrs_halfwidth"] = max(0.0, float(opt["qrs_halfwidth"]))
    opt["qrs_threshold_scale"] = min(1.0, max(0.0, float(opt["qrs_threshold_scale"])))
    opt["adaptive_window"] = max(0.0, float(opt["adaptive_window"]))
    opt["trials"] = max(2, int(opt["trials"]))
    try:
        opt["noise_width"] = float(opt["noise_width"])
        if not np.isfinite(opt["noise_width"]) or opt["noise_width"] < 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"EMD noise_width must be a number of 0 or more, got '{opt['noise_width']}'.") from None
    for key in ("max_imf", "seed"):
        if opt[key] is None:
            continue
        try:
            value = int(opt[key])
            if isinstance(opt[key], bool) or float(opt[key]) != value or \
                    (key == "max_imf" and value < -1):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"EMD {key} must be a whole number{' (-1 or blank: no limit)' if key == 'max_imf' else ''} "
                             f"or blank, got '{opt[key]}'.") from None
        opt[key] = None if key == "max_imf" and value == -1 else value
    return opt


def _mad_sigma(x):
    mad = np.median(np.abs(x - np.median(x)))
    sigma = float(mad) * _MAD_TO_SIGMA
    if sigma <= 0:
        sigma = float(np.std(x))
    return sigma


def _mean_frequency(x, fs):
    n = x.size
    if n < 8 or fs is None:
        return 0.0
    spectrum = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    total = spectrum.sum()
    if not np.isfinite(total) or total <= 0:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    return float((freqs * spectrum).sum() / total)


def _local_sigma(imf, window):
    n = imf.size
    if window <= 0 or n < 2 * window:
        return _mad_sigma(imf)
    step = max(1, window // 2)
    starts = np.arange(0, n - window + 1, step)
    centres = starts + window / 2.0
    sigmas = np.array([_mad_sigma(imf[s:s + window]) for s in starts])
    return np.interp(np.arange(n), centres, sigmas)


def _interval_threshold(imf, thr, mode="hard"):
    if np.ndim(thr) == 0 and thr <= 0:
        return imf
    sign = np.signbit(imf)
    starts = np.concatenate(([0], np.flatnonzero(sign[1:] != sign[:-1]) + 1))
    peaks = np.maximum.reduceat(np.abs(imf), starts)
    if np.ndim(thr) == 0:
        thr = np.full(peaks.size, float(thr))
    else:
        counts = np.diff(np.append(starts, imf.size))
        thr = np.add.reduceat(thr, starts) / counts

    gain = np.ones(peaks.size)
    below = peaks <= thr
    gain[below] = 0.0
    if mode == "soft":
        above = ~below
        gain[above] = (peaks[above] - thr[above]) / peaks[above]
    if gain.all():
        return imf

    labels = np.zeros(imf.size, dtype=np.intp)
    labels[starts[1:]] = 1
    np.cumsum(labels, out=labels)
    return imf * gain[labels]


def _detect_r_peaks(lead, fs):
    n = lead.size
    nyq = 0.5 * float(fs)
    if n < int(fs) or nyq <= 6:
        return np.empty(0, dtype=int)

    low, high = 5.0 / nyq, min(15.0, 0.9 * nyq) / nyq
    if not 0 < low < high < 1:
        return np.empty(0, dtype=int)
    try:
        sos = butter(2, [low, high], btype='band', output='sos')
        band = sosfiltfilt(sos, lead, padlen=min(3 * (2 * sos.shape[0] + 1), n - 1))
    except ValueError:
        return np.empty(0, dtype=int)

    energy = np.diff(band, prepend=band[0]) ** 2
    window = max(1, int(round(0.12 * fs)))
    envelope = np.convolve(energy, np.ones(window) / window, mode="same")

    height = 0.25 * np.percentile(envelope, 99)
    if not np.isfinite(height) or height <= 0:
        return np.empty(0, dtype=int)
    peaks, _ = find_peaks(envelope, height=height, distance=max(1, int(round(0.25 * fs))))
    if peaks.size == 0:
        return peaks

    search = max(1, int(round(0.05 * fs)))
    refined = []
    magnitude = np.abs(band)
    for p in peaks:
        lo, hi = max(0, p - search), min(n, p + search + 1)
        refined.append(lo + int(np.argmax(magnitude[lo:hi])))
    return np.unique(np.asarray(refined, dtype=int))


def _qrs_weight(peaks, n, fs, halfwidth, taper=0.02):
    weight = np.zeros(n)
    if peaks.size == 0 or halfwidth <= 0:
        return weight
    half = max(1, int(round(halfwidth * fs)))
    for p in peaks:
        weight[max(0, p - half):min(n, p + half + 1)] = 1.0
    ramp = max(1, int(round(taper * fs)))
    kernel = np.hanning(2 * ramp + 1)
    kernel /= kernel.sum()
    return np.clip(np.convolve(weight, kernel, mode="same"), 0.0, 1.0)


def _reference_lead(signals):
    best, best_score = 0, -np.inf
    for i, lead in enumerate(signals):
        spread = _mad_sigma(lead)
        if spread <= 0:
            continue
        score = float(np.percentile(np.abs(lead - np.median(lead)), 99.5)) / spread
        if score > best_score:
            best, best_score = i, score
    return best


def _make_decomposer(opt):
    max_imf = -1 if opt["max_imf"] in (None, 0) else int(opt["max_imf"])
    if opt["method"] == "emd":
        engine = EMD()
        return lambda x: engine.emd(x, max_imf=max_imf)

    from PyEMD import CEEMDAN, EEMD
    if opt["method"] == "eemd":
        engine = EEMD(trials=opt["trials"], noise_width=opt["noise_width"], parallel=False)
    else:
        engine = CEEMDAN(trials=opt["trials"], epsilon=opt["noise_width"], parallel=False)
    if opt["seed"] is not None and hasattr(engine, "noise_seed"):
        engine.noise_seed(int(opt["seed"]))
    return lambda x: engine(x, max_imf=max_imf)


def _denoise_lead(lead, fs, opt, decompose, qrs_weight):
    n = lead.size
    if n < 16 or np.ptp(lead) <= 0:
        return lead, [], 0

    imfs = np.atleast_2d(np.asarray(decompose(lead), dtype=np.float64))
    if imfs.size == 0 or imfs.shape[1] != n or imfs.shape[0] < 2:
        return lead, [], 0
    residue = lead - imfs.sum(axis=0)

    freqs = np.array([_mean_frequency(imf, fs) for imf in imfs])
    know_fs = fs is not None and fs > 0

    if opt["threshold"] > 0:
        noise_idx = set(np.flatnonzero(freqs > opt["noise_cut"]).tolist()) if know_fs else {0}
    else:
        noise_idx = set()

    drop_baseline = bool(opt["remove_baseline"]) and know_fs
    universal = np.sqrt(2.0 * np.log(n))
    sigma_window = int(round(opt["adaptive_window"] * fs)) if know_fs else 0

    out = np.zeros(n)
    thresholded, dropped = [], 0
    for k, imf in enumerate(imfs):
        if drop_baseline and freqs[k] < opt["baseline_cut"]:
            dropped += 1
            continue
        if k in noise_idx:
            thr = opt["threshold"] * _local_sigma(imf, sigma_window) * universal
            cleaned = _interval_threshold(imf, thr, opt["thresholding"])
            if cleaned is not imf:
                thresholded.append(k + 1)
                if opt["protect_qrs"] and qrs_weight is not None:
                    relaxed = _interval_threshold(imf, thr * opt["qrs_threshold_scale"],
                                                  opt["thresholding"])
                    cleaned = qrs_weight * relaxed + (1.0 - qrs_weight) * cleaned
            out += cleaned
        else:
            out += imf

    if drop_baseline and _mean_frequency(residue, fs) < opt["baseline_cut"]:
        dropped += 1
    else:
        out += residue
    return out, thresholded, dropped


def emd_denoise(ecg_signal, fs=None, options=None, log=None):
    if not _PYEMD_AVAILABLE:
        raise RuntimeError("PyEMD is not installed. Install it with 'pip install EMD-signal'.")

    opt = merge_emd_options(options)
    arr, was_1d = _as_2d(ecg_signal)
    n_leads, n = arr.shape
    if n == 0:
        return _restore(arr.copy(), was_1d)

    work = np.array(arr, dtype=np.float64, copy=True)
    for lead in work:
        bad = ~np.isfinite(lead)
        if bad.any():
            good = ~bad
            lead[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), lead[good]) \
                if good.any() else 0.0

    qrs_weight, n_beats = None, 0
    if opt["protect_qrs"] and fs:
        peaks = _detect_r_peaks(work[_reference_lead(work)], fs)
        n_beats = int(peaks.size)
        qrs_weight = _qrs_weight(peaks, n, fs, opt["qrs_halfwidth"])

    decompose = _make_decomposer(opt)
    out = np.empty_like(work)
    all_thresholded, total_dropped, failures = set(), 0, []
    for i in range(n_leads):
        try:
            lead_out, thresholded, dropped = _denoise_lead(work[i], fs, opt, decompose, qrs_weight)
        except Exception as exc:                      
            failures.append(f"lead {i + 1}: {exc}")
            lead_out = work[i]
            thresholded, dropped = [], 0
        out[i] = lead_out
        all_thresholded.update(thresholded)
        total_dropped += dropped

    if failures and len(failures) == n_leads:
        raise RuntimeError(f"EMD failed on every lead ({failures[0]})")

    if log is not None:
        if all_thresholded:
            detail = f"IMFs {sorted(all_thresholded)} thresholded"
        elif opt["threshold"] == 0:
            detail = "noise IMFs kept whole (threshold 0)"
        else:
            detail = "no noise IMF found"
        if opt["remove_baseline"]:
            detail += f", {total_dropped} baseline component(s) removed"
        if n_beats:
            detail += f", {n_beats} QRS window(s) protected"
        log(f"EMD ({opt['method']}): {detail}.")
        for failure in failures:
            log(f"EMD failed on {failure} — lead kept unfiltered.")

    if np.issubdtype(arr.dtype, np.floating) and arr.dtype != out.dtype:
        out = out.astype(arr.dtype)
    return _restore(out, was_1d)


# Resampling

def resample_signal(signal, original_fs, target_fs, method='fft'):
    arr, was_1d = _as_2d(signal)
    if not original_fs or not target_fs:
        raise ValueError("Both the original and the target sampling frequency must be known.")
    if original_fs == target_fs:
        return _restore(arr, was_1d)

    original_len = arr.shape[1]
    target_len = int(np.round(original_len * float(target_fs) / float(original_fs)))
    if target_len < 1:
        raise ValueError(f"Resampling {original_fs} Hz -> {target_fs} Hz leaves no samples.")
    if original_len == target_len:
        return _restore(arr, was_1d)

    if method.lower() == 'fft':
        if original_len < 2:
            return _restore(resample(arr, target_len, axis=1), was_1d)
        t_in = np.arange(original_len) / float(original_fs)
        t_out = np.arange(target_len) * (original_len / float(target_len)) / float(original_fs)
        first = arr[:, :1]
        slope = (arr[:, -1:] - first) / t_in[-1]
        out = resample(arr - first - slope * t_in, target_len, axis=1) + first + slope * t_out
        return _restore(out.astype(arr.dtype, copy=False), was_1d)
    elif method.lower() == 'interpolation':
        src = np.linspace(0.0, original_len - 1, original_len)
        dst = np.linspace(0.0, original_len - 1, target_len)
        out = np.empty((arr.shape[0], target_len), dtype=arr.dtype)
        for i in range(arr.shape[0]):
            out[i] = np.interp(dst, src, arr[i])
        return _restore(out, was_1d)
    else:
        raise ValueError(f"Unknown resampling method '{method}'. Choose from 'fft' or 'interpolation'.")


# Pipeline

PIPELINE_STEPS = ("notch", "bandpass", "wavelet", "emd")


def preprocess_signals(signals, pipeline, fs, notch_freq, emd_options=None, log=None):
    if fs is None:
        raise ValueError("Sampling frequency (fs) must be known for processing.")

    for step in pipeline:
        if step == 'notch':
            signals = notch_filter(signals, fs=fs, notch_freq=notch_freq)
        elif step == 'bandpass':
            signals = bandpass_filter(signals, fs=fs)
        elif step == 'wavelet':
            signals = wavelet_baseline_removal(signals, level=9, fs=fs)
        elif step == 'emd':
            if not _PYEMD_AVAILABLE:
                if log is not None:
                    log("PyEMD not found. Skipping EMD step. Install it with 'pip install EMD-signal'.")
                continue
            signals = emd_denoise(signals, fs=fs, options=emd_options, log=log)
        else:
            raise ValueError(f"Unknown pre-processing step '{step}'. "
                             f"Choose from {', '.join(PIPELINE_STEPS)}.")
    return signals
