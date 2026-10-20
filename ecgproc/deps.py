import re
import sys
import importlib
import importlib.util
from importlib import metadata

_checked = False


def _version_tuple(text):
    """'2.6.0+cpu' -> (2, 6, 0), '2.5rc1' -> (2, 5): the leading digits of each part."""
    parts = []
    for piece in str(text).split('+')[0].split('.')[:3]:
        digits = re.match(r'\d+', piece)
        if not digits:
            break
        parts.append(int(digits.group()))
    return tuple(parts)


def _installed_version(pip_name):
    try:
        return metadata.version(pip_name)
    except metadata.PackageNotFoundError:
        return None


def check_dependencies(import_modules=True):
    """
    Checks for required packages and provides instructions if they are missing,
    installed but not importable, or too old.
    """
    global _checked
    if _checked:
        return
    # List of (pip_package_name, import_name, minimum version or None)
    packages = [
        ('numpy', 'numpy', None),
        ('pandas', 'pandas', None),
        ('tqdm', 'tqdm', None),
        ('scipy', 'scipy', None),
        ('pydicom', 'pydicom', None),
        ('PyWavelets', 'pywt', None),
        ('lxml', 'lxml', None),
        ('EMD-signal', 'PyEMD', None),
        ('wfdb', 'wfdb', None),
        ('matplotlib', 'matplotlib', None),
        ('neurokit2', 'neurokit2', None),
        # Safe weight loading (ecgxai.registry) needs PyTorch 2.6 or later
        ('torch', 'torch', (2, 6)),
    ]
    optional = {'lxml': "recovering malformed vendor XML",
                'EMD-signal': "the 'emd' pre-processing step"}

    print("Checking for required packages...")
    missing_packages, broken_packages, old_packages = [], [], []
    for pip_name, import_name, minimum in packages:
        try:
            if import_modules:
                module = importlib.import_module(import_name)
            else:
                module = None
                if importlib.util.find_spec(import_name) is None:
                    raise ModuleNotFoundError(f"No module named '{import_name}'")
        except Exception as e:      # ImportError, or whatever a broken install raises
            installed = _installed_version(pip_name)
            if installed is None:
                missing_packages.append(pip_name)
            else:
                broken_packages.append((pip_name, installed, import_name, e))
            continue
        if minimum:
            version = getattr(module, '__version__', None) or _installed_version(pip_name) or ''
            if _version_tuple(version) < minimum:
                old_packages.append((pip_name, version, '.'.join(map(str, minimum))))

    for pkg in [p for p in missing_packages if p in optional]:
        missing_packages.remove(pkg)
        print(f"Note: {pkg} is not installed; it is only needed for {optional[pkg]}.")
    for entry in [b for b in broken_packages if b[0] in optional]:
        broken_packages.remove(entry)
        pkg, version, import_name, err = entry
        print(f"Note: {pkg} {version} is installed but 'import {import_name}' fails "
              f"({type(err).__name__}: {err}); it is only needed for {optional[pkg]}.")
        if pkg == 'EMD-signal' and _installed_version('pyemd'):
            print("    The unrelated 'pyemd' package shares its folder; to fix it run:\n"
                  "        pip uninstall -y pyemd EMD-signal && pip install EMD-signal")

    if not (missing_packages or broken_packages or old_packages):
        print("All required packages are available.\n")
        _checked = True
        return

    if missing_packages:
        print("\nERROR: The following required packages are not installed:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
    if broken_packages:
        print("\nERROR: The following packages are installed but cannot be imported:")
        for pkg, version, import_name, err in broken_packages:
            print(f"  - {pkg} {version}: 'import {import_name}' fails "
                  f"({type(err).__name__}: {err})")
            if pkg == 'EMD-signal' and _installed_version('pyemd'):
                print("    The unrelated 'pyemd' package is installed as well and shares its folder.\n"
                      "    Remove both, then reinstall EMD-signal:\n"
                      "        pip uninstall -y pyemd EMD-signal && pip install EMD-signal")
    if old_packages:
        print("\nERROR: The following packages are too old:")
        for pkg, version, need in old_packages:
            print(f"  - {pkg} {version} (needs {need} or later)")
    print("\nInstall them from the requirements file that ships with the repository,")
    print("preferably into a fresh virtual environment (Python 3.10+):")
    print("    pip install -r requirements.txt")
    print("\nThe script will now exit.")
    sys.exit(1)
