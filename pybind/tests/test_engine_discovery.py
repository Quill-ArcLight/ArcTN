"""Engine selection uses the platform-specific file in the installed package."""

import os

import pytest

from arctn import _engine


@pytest.mark.parametrize("platform,filename", [
    ("linux", "libarctn_engine.so"),
    ("darwin", "libarctn_engine.dylib"),
    ("win32", "arctn_engine.dll"),
])
def test_platform_library_selection(tmp_path, monkeypatch, platform, filename):
    package = tmp_path / "arctn"
    library_dir = package / "_lib"
    library_dir.mkdir(parents=True)
    monkeypatch.setattr(_engine, "__file__", str(package / "_engine.py"))
    monkeypatch.setattr(_engine.sys, "platform", platform)
    monkeypatch.delenv("ARCTN_ENGINE_LIBRARY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))

    # Same-name files in the working directory or PATH are not candidates.
    (tmp_path / filename).touch()
    filenames = {"libarctn_engine.so", "libarctn_engine.dylib", "arctn_engine.dll"}
    for other in filenames - {filename}:
        (library_dir / other).touch()
    _engine._configure_bundled_engine()
    assert "ARCTN_ENGINE_LIBRARY" not in os.environ

    library = library_dir / filename
    library.touch()
    _engine._configure_bundled_engine()
    assert os.environ["ARCTN_ENGINE_LIBRARY"] == str(library)


def test_unknown_platform_does_not_select_engine(tmp_path, monkeypatch):
    package = tmp_path / "arctn"
    library_dir = package / "_lib"
    library_dir.mkdir(parents=True)
    for filename in ("libarctn_engine.so", "libarctn_engine.dylib", "arctn_engine.dll"):
        (library_dir / filename).touch()
    monkeypatch.setattr(_engine, "__file__", str(package / "_engine.py"))
    monkeypatch.setattr(_engine.sys, "platform", "unsupported")
    monkeypatch.delenv("ARCTN_ENGINE_LIBRARY", raising=False)
    _engine._configure_bundled_engine()
    assert "ARCTN_ENGINE_LIBRARY" not in os.environ
