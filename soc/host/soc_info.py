#!/usr/bin/env python3
"""Print the resolved SoC/VTA contract used by host-side packaging tools."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csr-json", type=Path, required=True)
    parser.add_argument("--vta-manifest", type=Path, required=True)
    args = parser.parse_args()
    csr = json.loads(args.csr_json.read_text())
    manifest = json.loads(args.vta_manifest.read_text())
    constants = csr.get("constants", {})
    result = {
        "vta_variant": manifest["variant"],
        "vta_rtl_sha256": manifest["rtl_sha256"],
        "vcr_base": constants.get("vta_vcr_base"),
        "memory_data_width": constants.get("vta_memory_data_width"),
        "memory_protocol": manifest["memory_protocol"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
