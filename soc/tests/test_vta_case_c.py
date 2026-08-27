import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "vta_case_c.py"
SPEC = importlib.util.spec_from_file_location("vta_case_c", MODULE_PATH)
VTA_CASE_C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VTA_CASE_C)


class VTACaseCTest(unittest.TestCase):
    def test_interface_alignment_tracks_memory_width(self):
        for log_bus_width, beat_bytes in ((5, 4), (6, 8), (7, 16)):
            manifest = {"config": {"LOG_BUS_WIDTH": log_bus_width}}
            self.assertEqual(VTA_CASE_C.alignment_for("insn", manifest), 16 * beat_bytes)
            for name in ("uop", "inp", "wgt", "acc", "out"):
                self.assertEqual(VTA_CASE_C.alignment_for(name, manifest), 8)

    def test_ai_sections(self):
        self.assertEqual(VTA_CASE_C.section_for("insn"), ".rodata_ai")
        self.assertEqual(VTA_CASE_C.section_for("wgt"), ".rodata_ai")
        self.assertEqual(VTA_CASE_C.section_for("inp"), ".data_ai.init")
        self.assertEqual(VTA_CASE_C.section_for("out"), ".data_ai.out")


if __name__ == "__main__":
    unittest.main()
