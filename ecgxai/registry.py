import importlib
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from ecgproc.constants import FINAL_LEADS


def app_root():
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    task: str
    weights: str
    architecture: str = "ECGResidualModelBinary"
    arch: tuple = (("num_layers", 3), ("width_scale", 1.0), ("drop_path_rate", 0.2),
                   ("dropout_rate", 0.5), ("activation_choice", "relu"))
    labels: tuple = ("No Ischemia", "Ischemia")
    positive_class: int = 1
    threshold: float = 0.564
    fs: int = 500
    window_s: float = 3.0
    leads: tuple = tuple(FINAL_LEADS)
    age_mean: float = 56.47
    age_std: float = 18.35
    sex_codes: tuple = (("Male", 1.0), ("Female", 0.0))
    gradcam_layer: str = "layer3"

    @property
    def window_samples(self):
        return int(round(self.fs * self.window_s))

    @property
    def weights_path(self):
        return app_root() / self.weights

    def sex_code(self, sex):
        return dict(self.sex_codes)[sex]

    def label(self, cls):
        return self.labels[int(cls)]


MODELS = {
    "cardiac-ischemia-xnet": ModelSpec(
        key="cardiac-ischemia-xnet",
        name="Cardiac-Ischemia-XNet",
        task="Ischemia vs. no ischemia from 3 s of 12-lead ECG at 500 Hz, plus age and sex",
        weights="models/finetuned_ischemia_robust_model.pth",
    ),
}
DEFAULT_MODEL = "cardiac-ischemia-xnet"


def get_spec(name=None):
    wanted = str(name or DEFAULT_MODEL).strip().lower()
    for spec in MODELS.values():
        if wanted in (spec.key, spec.name.lower()):
            return spec
    raise ValueError(f"Unknown model '{name}'. Choose from {', '.join(MODELS)}.")


class LoadedModel:

    def __init__(self, spec, module, device, info):
        self.spec = spec
        self.module = module
        self.device = device
        self.info = info
        self.lock = threading.RLock()


_CACHE = {}
_CACHE_LOCK = threading.Lock()


def resolve_device(device="cpu"):
    import torch

    wanted = str(device or "cpu").strip().lower()
    mps = getattr(torch.backends, "mps", None)
    has_mps = bool(mps is not None and mps.is_available())
    if wanted == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if has_mps else "cpu")
    if wanted == "cuda" and not torch.cuda.is_available():
        raise ValueError("device = cuda, but PyTorch sees no CUDA device. Use cpu or auto.")
    if wanted == "mps" and not has_mps:
        raise ValueError("device = mps, but PyTorch sees no Apple GPU. Use cpu or auto.")
    if wanted not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unknown device '{device}'. Choose from cpu, cuda, mps, auto.")
    return torch.device(wanted)


def _read_checkpoint(path):
    import numpy as np
    import torch

    version = tuple(int(p) for p in re.findall(r"\d+", torch.__version__)[:2])
    if version < (2, 6):
        raise RuntimeError(f"PyTorch {torch.__version__} is too old to load the weights safely; "
                           "install 2.6 or later (pip install -U torch).")
    try:
        multiarray = importlib.import_module("numpy._core.multiarray")
    except ImportError:
        multiarray = importlib.import_module("numpy.core.multiarray")
    allowed = [np.dtype, (multiarray.scalar, "numpy.core.multiarray.scalar"), multiarray.scalar]
    try:
        allowed.append(type(np.dtype(np.float64)))
    except TypeError:
        pass
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def _state_dict(checkpoint):
    state = checkpoint
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    state = dict(state)
    state.pop("n_averaged", None)
    return state


def _build_residual_binary(spec, checkpoint):
    from ecgxai.architectures import ECGResidualModelBinary

    model = ECGResidualModelBinary(n_classes=len(spec.labels), **dict(spec.arch))
    model.load_state_dict(_state_dict(checkpoint), strict=True)
    return model


_BUILDERS = {"ECGResidualModelBinary": _build_residual_binary}


def _scalar(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_model(name=None, device="cpu"):
    spec = get_spec(name)
    dev = resolve_device(device)
    key = (spec.key, str(dev))
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
        path = spec.weights_path
        if not path.is_file():
            raise FileNotFoundError(f"Model weights for {spec.name} not found at {path}.")
        if path.stat().st_size < 4096:
            raise RuntimeError(f"{path} is only {path.stat().st_size} bytes; it looks like a Git LFS "
                               "pointer rather than the weights. Fetch the real file (git lfs pull).")
        try:
            checkpoint = _read_checkpoint(path)
            module = _BUILDERS[spec.architecture](spec, checkpoint)
        except Exception as e:
            raise RuntimeError(f"Could not load {spec.name} from {path}: {e}") from e
        module.to(dev)
        module.eval()
        ck = checkpoint if isinstance(checkpoint, dict) else {}
        n_averaged = _scalar(ck.get("n_averaged"))
        info = {
            "epoch": ck.get("epoch"),
            "val_auc": _scalar(ck.get("auc")),
            "val_best_threshold": _scalar(ck.get("best_threshold")),
            "activation": dict(spec.arch).get("activation_choice"),
            "dropout": dict(spec.arch).get("dropout_rate"),
            "swa_snapshots": int(n_averaged) if n_averaged is not None else None,
            "parameters": int(sum(p.numel() for p in module.parameters())),
        }
        loaded = LoadedModel(spec, module, dev, info)
        _CACHE[key] = loaded
        return loaded


def describe(loaded):
    info = loaded.info
    bits = [f"{loaded.spec.name} on {loaded.device}"]
    if info.get("epoch") is not None:
        bits.append(f"checkpoint epoch {info['epoch']}")
    if info.get("val_auc") is not None:
        bits.append(f"validation AUC {info['val_auc']:.3f}")
    if info.get("swa_snapshots") is not None:
        bits.append(f"SWA average of {info['swa_snapshots']} snapshots")
    bits.append(f"{info['parameters']:,} parameters")
    return " · ".join(bits)