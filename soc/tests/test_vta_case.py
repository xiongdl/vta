import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "vta_case.py"
SPEC = importlib.util.spec_from_file_location("vta_case", MODULE_PATH)
VTA_CASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VTA_CASE)


class VTACaseTest(unittest.TestCase):
    def test_interface_alignment_tracks_memory_width(self):
        for log_bus_width, beat_bytes in ((5, 4), (6, 8), (7, 16)):
            config = {"LOG_BUS_WIDTH": log_bus_width}
            self.assertEqual(VTA_CASE.interface_alignment("insn", config), 16 * beat_bytes)
            for name in ("uop", "inp", "wgt", "acc", "out"):
                self.assertEqual(VTA_CASE.interface_alignment(name, config), 8)

    def make_case(self, directory: Path):
        (directory / "insn.bin").write_bytes(bytes(16))
        (directory / "out.bin").write_bytes(bytes(64))
        sections = {}
        for name, address in (("insn", 0x40010000), ("out", 0x40020000)):
            path = directory / f"{name}.bin"
            sections[name] = {"file": path.name, "address": address,
                              "size": path.stat().st_size,
                              "sha256": VTA_CASE.sha256(path), "alignment": 64}
        (directory / "manifest.json").write_text(json.dumps({
            "format": "vta-soc-case-v1", "insn_count": 1, "sections": sections}))

    def test_validate_hashes_and_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_case(directory)
            VTA_CASE.validate(directory)

    def test_rejects_section_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_case(directory)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["sections"]["out"]["address"] = 0x40010008
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "overlap"):
                VTA_CASE.validate(directory)


if __name__ == "__main__":
    unittest.main()
