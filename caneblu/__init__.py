# __init__.py
"""
caneblu package initializer.

Public submodules:
- composite_indices
- geocoding
- harvester
- satellite_image_utils
- utils
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version

__version__ = "0.1.0"
__author__ = 'Edoardo Carlesi'
__credits__ = 'Cane colorato ma tendenzialmente blu'

# List of lazily-loadable submodules
_SUBMODULES = {
    "composite_indices",
    "geocoding",
    "harvester",
    "satellite_image_utils",
    "utils",
    "evalscript_generator_utils",
    "open_meteo_api",
}

def __getattr__(name: str):
    # Lazy import submodules on first attribute access
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # Make submodules appear in dir() and help()
    return sorted(list(globals().keys()) + list(_SUBMODULES))


def _detect_version() -> str:
    # Try to obtain version from installed package metadata
    # First try the current package name, then common variants
    candidates = {__name__, "caneblu"}

    for candidate in candidates:
        try:
            return _pkg_version(candidate)
        except PackageNotFoundError:
            continue
    return "0.0.0"


__version__ = _detect_version()
