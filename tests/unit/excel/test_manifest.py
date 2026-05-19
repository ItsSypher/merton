"""Manifest XML generation tests."""

from __future__ import annotations

from xml.etree import ElementTree as ET

from merton.excel.manifest import (
    deterministic_add_in_id,
    render_manifest,
    write_manifest,
)


class TestManifest:
    def test_render_is_valid_xml(self) -> None:
        xml = render_manifest()
        root = ET.fromstring(xml)
        # OfficeApp is the root element.
        assert root.tag.endswith("OfficeApp")

    def test_contains_required_elements(self) -> None:
        xml = render_manifest(base_url="http://localhost:9001")
        for needle in (
            "<Id>",
            "<DisplayName ",
            "<Description ",
            "ReadWriteDocument",
            "http://localhost:9001",
            "MERTON",  # the function namespace
        ):
            assert needle in xml, f"missing {needle!r}"

    def test_deterministic_id(self) -> None:
        a = deterministic_add_in_id("http://localhost:8000")
        b = deterministic_add_in_id("http://localhost:8000")
        c = deterministic_add_in_id("http://localhost:9000")
        assert a == b
        assert a != c

    def test_custom_version(self) -> None:
        xml = render_manifest(version="2.3.4.5")
        assert "<Version>2.3.4.5</Version>" in xml

    def test_explicit_add_in_id(self) -> None:
        xml = render_manifest(add_in_id="11111111-2222-3333-4444-555555555555")
        assert "11111111-2222-3333-4444-555555555555" in xml

    def test_write_manifest_to_disk(self, tmp_path) -> None:
        path = write_manifest(tmp_path / "manifest.xml", base_url="http://localhost:9100")
        assert path.exists()
        content = path.read_text()
        assert "http://localhost:9100" in content
