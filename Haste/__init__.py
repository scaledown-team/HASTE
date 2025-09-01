"""haste: lightweight entry point."""

from importlib.metadata import PackageNotFoundError, version

# Force version to match setup.py
__version__ = "0.2.1"

# Commented out for now to force use of our hardcoded version
# try:
#     __version__ = version("haste")
# except PackageNotFoundError:
#     __version__ = "0.2.1"

# Public API facade (re-export)
from .api import build_payload_from_repo  # noqa: F401
from .api import build_structural_context_from_source, select_from_file

__all__ = [
    "__version__",
    "select_from_file",
    "build_payload_from_repo",
    "build_structural_context_from_source",
]


