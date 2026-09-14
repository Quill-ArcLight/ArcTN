"""Select the optional engine binary shipped inside this Python package."""

import os
from pathlib import Path
import sys


def _configure_bundled_engine():
    # An explicit setting, including an empty or invalid path, takes priority.
    # The native loader reports errors when planning is requested.
    if "ARCTN_ENGINE_LIBRARY" in os.environ:
        return

    filename = {
        "linux": "libarctn_engine.so",
        "darwin": "libarctn_engine.dylib",
        "win32": "arctn_engine.dll",
    }.get(sys.platform)
    if filename is None:
        return

    library = Path(__file__).resolve().parent / "_lib" / filename
    if library.is_file():
        os.environ.setdefault("ARCTN_ENGINE_LIBRARY", str(library))
