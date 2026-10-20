# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ECG-XAI-Visualizer standalone builds."""

import sys

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

datas = [
    ("config.ini", "."),
    ("models/finetuned_ischemia_robust_model.pth", "models"),
]
datas += collect_data_files("neurokit2")

hidden_imports = [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    "matplotlib.font_manager",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtCore",
    "scipy.interpolate",
    "scipy.signal",
    "scipy.ndimage",
    "scipy.io",
    "pydicom",
    "pywt",
    "PyEMD",
    "wfdb",
    "lxml",
    "lxml.etree",
    "neurokit2",
    "torch",
    # imported inside functions, so name them explicitly
    "ecgxai.analyze",
    "ecgxai.architectures",
    "ecgxai.explain",
    "ecgxai.inference",
    "ecgxai.segments",
]

a = Analysis(
    ["ui.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The model runs on the CPU build of PyTorch (see build.yml); nothing here
    # needs the GPU toolchains or the notebook stack.
    excludes=["tkinter", "IPython", "jupyter", "notebook", "torchvision", "torchaudio",
              "tensorboard", "triton"],
    noarchive=False,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name="ECG-XAI-Visualizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX corrupts some PyTorch libraries
    console=False,
    disable_windowed_traceback=False,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="ECG-XAI-Visualizer",
    strip=False,
    upx=False,
    upx_exclude=[],
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="ECG-XAI-Visualizer.app",
        icon=None,
        bundle_identifier="io.github.cebida.ecg-xai-visualizer",
        info_plist={"NSHighResolutionCapable": True},
    )
