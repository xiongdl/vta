import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "soc_config.py"
SPEC = importlib.util.spec_from_file_location("soc_config", MODULE_PATH)
SOC_CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOC_CONFIG)


class SoCConfigTest(unittest.TestCase):
    def base(self):
        path = Path(__file__).parents[1] / "config" / "sim_default.json"
        return SOC_CONFIG.load_config(path)

    def test_default_structure_without_ip(self):
        cfg = SOC_CONFIG.validate(self.base(), require_ip=False)
        self.assertEqual(cfg["vta"]["memory_protocol"], "axi4")
        self.assertEqual(cfg["vta"]["memory_data_width"], 64)

    def test_rejects_overlap(self):
        cfg = self.base()
        cfg["sram"]["base"] = cfg["rom"]["base"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            SOC_CONFIG.validate(cfg, require_ip=False)

    def test_rejects_unsupported_memory_width(self):
        cfg = self.base()
        cfg["vta"]["memory_data_width"] = 128
        with self.assertRaisesRegex(ValueError, "VTA memory"):
            SOC_CONFIG.validate(cfg, require_ip=False)

    def test_generates_shared_headers(self):
        cfg = SOC_CONFIG.validate(self.base(), require_ip=False)
        with tempfile.TemporaryDirectory() as tmp:
            SOC_CONFIG.emit(cfg, Path(tmp))
            self.assertIn("VTA_VCR_BASE", (Path(tmp) / "soc_config.h").read_text())
            self.assertTrue((Path(tmp) / "soc_config.resolved.json").is_file())

    def test_frozen_ip_hash_matches_manifest(self):
        cfg = SOC_CONFIG.validate(self.base(), require_ip=True)
        ip_dir = cfg["_vta_ip_dir"]
        manifest = json.loads((ip_dir / "manifest.json").read_text())
        digest = hashlib.sha256((ip_dir / "rtl" / "vta.v").read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["rtl_sha256"])


if __name__ == "__main__":
    unittest.main()
