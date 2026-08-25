#!/usr/bin/env python3
"""Import an already-generated VTA RTL file into an immutable SoC IP package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


VARIANTS = {
    "apb32-ahb32": ("VTAShellAPBAHB", "apb4", 32, "ahb-lite", 32),
    "apb32-ahb64": ("VTAShellAPBAHB", "apb4", 32, "ahb-lite", 64),
    "apb32-axi32": ("VTAShellAPB", "apb4", 32, "axi4", 32),
    "apb32-axi64": ("VTAShellAPB", "apb4", 32, "axi4", 64),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--rtl", type=Path, required=True)
    parser.add_argument("--vta-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generator", default="vta.APBHostDefaultDe10Config")
    args = parser.parse_args()
    if not args.rtl.is_file() or not args.vta_config.is_file():
        parser.error("--rtl and --vta-config must identify existing files")
    top, host_protocol, host_width, mem_protocol, mem_width = VARIANTS[args.variant]
    text = args.rtl.read_text(errors="replace")
    if re.search(rf"\bmodule\s+{re.escape(top)}\b", text) is None:
        parser.error(f"RTL does not contain expected top module {top}")
    config = json.loads(args.vta_config.read_text())
    actual_width = 1 << int(config["LOG_BUS_WIDTH"])
    if actual_width != mem_width:
        parser.error(f"VTA config bus is {actual_width}-bit, variant requires {mem_width}-bit")

    out = args.output_root / args.variant
    rtl_dir = out / "rtl"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    rtl_out = rtl_dir / "vta.v"
    shutil.copyfile(args.rtl, rtl_out)
    shutil.copyfile(args.vta_config, out / "vta_config.json")
    digest = hashlib.sha256(rtl_out.read_bytes()).hexdigest()
    config_digest = hashlib.sha256(args.vta_config.read_bytes()).hexdigest()
    repo = Path(__file__).resolve().parents[2]
    try:
        source_commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unknown"
    manifest = {
        "schema_version": 1,
        "variant": args.variant,
        "top": top,
        "host_protocol": host_protocol,
        "host_data_width": host_width,
        "memory_protocol": mem_protocol,
        "memory_data_width": mem_width,
        "memory_address_width": 32,
        "rtl_sha256": digest,
        "vta_config_sha256": config_digest,
        "source_commit": source_commit,
        "generator": args.generator,
        "source_policy": "pre-generated RTL; SoC builds must not invoke Chisel"
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "filelist.f").write_text("rtl/vta.v\n")
    (out / "rtl.sha256").write_text(f"{digest}  rtl/vta.v\n")
    print(out)


if __name__ == "__main__":
    main()
