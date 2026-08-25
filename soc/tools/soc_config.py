#!/usr/bin/env python3
"""Validate Mini VTA SoC configuration and generate shared constants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ALLOWED_MEMORY = {
    (protocol, width)
    for protocol in ("ahb-lite", "axi4")
    for width in (32, 64, 128)
}


def number(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    raise ValueError(f"expected integer or integer string, got {value!r}")


def _merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path, parents=()):
    path = path.resolve()
    if path in parents:
        raise ValueError(f"configuration inheritance cycle at {path}")
    data = json.loads(path.read_text())
    parent = data.pop("extends", None)
    if parent is not None:
        data = _merge(load_config(path.parent / parent, parents + (path,)), data)
    data["_config_path"] = path.resolve()
    return data


def region(name, section):
    base, size = number(section["base"]), number(section["size"])
    if size <= 0 or size & (size - 1):
        raise ValueError(f"{name}.size must be a positive power of two")
    if base % size:
        raise ValueError(f"{name}.base must be aligned to its size")
    return name, base, size


def validate(data, require_ip=True):
    if data.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if data["soc"]["address_width"] != 32:
        raise ValueError("the first release supports a 32-bit address space")
    if data["vta"]["host_protocol"] not in {"apb4", "ahb-lite"} or data["vta"]["host_data_width"] != 32:
        raise ValueError("the VTA system host interface must be APB4/AHB-Lite 32-bit")
    if data["vta"]["vcr_frontend"] not in {"apb32", "ahb32"}:
        raise ValueError("vta.vcr_frontend must be apb32 or ahb32")
    mem = (data["vta"]["memory_protocol"], data["vta"]["memory_data_width"])
    if mem not in ALLOWED_MEMORY:
        raise ValueError("VTA memory must be ahb-lite/axi4 with 32/64/128-bit data")
    if data["sram"]["data_width"] not in {32, 64, 128}:
        raise ValueError("sram.data_width must be 32, 64 or 128")
    if data["ddr"]["data_width"] not in {32, 64, 128}:
        raise ValueError("ddr.data_width must be 32, 64 or 128")

    regions = [region("rom", data["rom"]), region("sram", data["sram"])]
    regions.append(("vta-vcr", number(data["vta"]["vcr_base"]), number(data["vta"]["vcr_size"])))
    if data["ddr"]["enabled"]:
        regions.append(region("ddr", data["ddr"]))
    for index, (name_a, base_a, size_a) in enumerate(regions):
        if base_a + size_a > 1 << 32:
            raise ValueError(f"{name_a} exceeds the 32-bit address space")
        for name_b, base_b, size_b in regions[index + 1 :]:
            if max(base_a, base_b) < min(base_a + size_a, base_b + size_b):
                raise ValueError(f"address regions {name_a} and {name_b} overlap")

    config_dir = data["_config_path"].parent
    ip_dir = (config_dir / data["vta"]["ip_dir"]).resolve()
    if require_ip:
        manifest_path = ip_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"frozen VTA manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        rtl_path = ip_dir / "rtl" / "vta.v"
        rtl_digest = hashlib.sha256(rtl_path.read_bytes()).hexdigest()
        if rtl_digest != manifest.get("rtl_sha256"):
            raise ValueError(f"frozen VTA RTL hash mismatch: {rtl_path}")
        vta_config_path = ip_dir / "vta_config.json"
        config_digest = hashlib.sha256(vta_config_path.read_bytes()).hexdigest()
        if config_digest != manifest.get("vta_config_sha256"):
            raise ValueError(f"frozen VTA config hash mismatch: {vta_config_path}")
        # AHB32 is a system-side bridge in front of the frozen native APB32
        # VCR port; the frozen IP manifest therefore remains APB4.
        expected = {
            "host_protocol": "apb4",
            "host_data_width": 32,
            "memory_protocol": data["vta"]["memory_protocol"],
            "memory_data_width": data["vta"]["memory_data_width"],
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"VTA manifest {key}={manifest.get(key)!r}, expected {value!r}")
    data["_vta_ip_dir"] = ip_dir
    return data


def emit(data, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    values = {
        "SOC_CLOCK_HZ": data["soc"]["clock_hz"],
        "ROM_BASE": number(data["rom"]["base"]),
        "ROM_SIZE": number(data["rom"]["size"]),
        "SRAM_BASE": number(data["sram"]["base"]),
        "SRAM_SIZE": number(data["sram"]["size"]),
        "SRAM_DATA_WIDTH": data["sram"]["data_width"],
        "DDR_BASE": number(data["ddr"]["base"]),
        "DDR_SIZE": number(data["ddr"]["size"]),
        "VTA_VCR_BASE": number(data["vta"]["vcr_base"]),
        "VTA_VCR_SIZE": number(data["vta"]["vcr_size"]),
        "VTA_MEMORY_DATA_WIDTH": data["vta"]["memory_data_width"],
    }
    header = ["/* Generated; do not edit. */", "#pragma once"]
    verilog = ["// Generated; do not edit."]
    for key, value in values.items():
        header.append(f"#define {key} 0x{value:x}u" if key.endswith(("BASE", "SIZE")) else f"#define {key} {value}u")
        verilog.append(f"`define {key} {value}")
    (output / "soc_config.h").write_text("\n".join(header) + "\n")
    (output / "soc_config.vh").write_text("\n".join(verilog) + "\n")
    clean = {key: value for key, value in data.items() if not key.startswith("_")}
    (output / "soc_config.resolved.json").write_text(json.dumps(clean, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "generate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, required=True)
        cmd.add_argument("--allow-missing-ip", action="store_true")
        if name == "generate":
            cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = validate(load_config(args.config), require_ip=not args.allow_missing_ip)
    if args.command == "generate":
        emit(config, args.output)
    print(f"configuration valid: {args.config}")


if __name__ == "__main__":
    main()
