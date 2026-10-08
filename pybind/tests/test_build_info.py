"""Build identity belongs to the loaded public extension, not the engine."""

from arctn import arctn_py


def test_build_info_reports_public_compile_features():
    info = arctn_py.build_info()
    assert {
        "version", "commit", "source_state", "target_arch", "target_os",
        "target_pointer_width", "features",
    } <= info.keys()
    assert {"mt", "integer-order-dp", "integer-tree-cost"} <= info["features"].keys()
    for name in ("mt", "integer-order-dp", "integer-tree-cost"):
        assert type(info["features"][name]) is bool


def test_build_info_does_not_inspect_external_engine(monkeypatch, tmp_path):
    before = arctn_py.build_info()
    monkeypatch.setenv("ARCTN_ENGINE_LIBRARY", str(tmp_path / "missing-engine"))
    assert arctn_py.build_info() == before
    monkeypatch.delenv("ARCTN_ENGINE_LIBRARY")
    assert arctn_py.build_info() == before
