#!/usr/bin/env python3
"""Pack an FSIM/TSIM raw memory dump into a relocatable LiteX VTA case."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path


LOAD, STORE = 0, 1
DEFAULT_BASES = {
    "insn": 0x40010000,
    "uop": 0x40020000,
    "inp": 0x40030000,
    "wgt": 0x40040000,
    "acc": 0x40080000,
    "out": 0x400D0000,
}


def align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_abi(repo: Path, output: Path) -> None:
    raw_defs = subprocess.check_output([
        "python3", str(repo / "config/vta_config.py"),
        f"--use-cfg={repo / 'config/vta_config.json'}", "--defs"], text=True)
    definitions = [item for item in shlex.split(raw_defs)
                   if not item.startswith("-DVTA_TARGET=") and not item.startswith("-DVTA_HW_VER=")]
    subprocess.run(["/usr/bin/clang++", "-std=c++17", f"-I{repo / 'include'}",
                    *definitions, str(repo / "soc/tools/vta_case_abi.cc"),
                    "-o", str(output)], check=True)


def allocation_for(manifest: dict, address: int) -> dict:
    for allocation in manifest["allocations"]:
        if allocation["phy_addr"] <= address < allocation["phy_addr"] + allocation["size"]:
            return allocation
    raise ValueError(f"address 0x{address:x} is outside all dumped allocations")


def discover_raw(raw: Path) -> Path:
    if (raw / "before_manifest.json").is_file():
        return raw
    runs = sorted(path for path in raw.glob("run_*") if (path / "before_manifest.json").is_file())
    if not runs:
        raise ValueError(f"no raw VTA case found under {raw}")
    return runs[-1]


def pack(raw: Path, output: Path, repo: Path) -> None:
    raw = discover_raw(raw)
    before = json.loads((raw / "before_manifest.json").read_text())
    after = json.loads((raw / "after_manifest.json").read_text())
    config = json.loads((repo / "config/vta_config.json").read_text())
    batch = 1 << config["LOG_BATCH"]
    block_in = 1 << config.get("LOG_BLOCK_IN", config["LOG_BLOCK"])
    block_out = 1 << config.get("LOG_BLOCK_OUT", config["LOG_BLOCK"])
    elem = {
        "uop": 4,
        "wgt": (1 << config["LOG_WGT_WIDTH"]) * block_in * block_out // 8,
        "inp": (1 << config["LOG_INP_WIDTH"]) * batch * block_in // 8,
        "acc": (1 << config["LOG_ACC_WIDTH"]) * batch * block_out // 8,
        "out": (1 << config.get("LOG_OUT_WIDTH", config["LOG_INP_WIDTH"])) * batch * block_out // 8,
    }
    raw_abi = before.get("abi")
    calculated_abi = {"insn_bytes": 16, "uop_bytes": elem["uop"],
                      "inp_elem_bytes": elem["inp"], "wgt_elem_bytes": elem["wgt"],
                      "acc_elem_bytes": elem["acc"], "out_elem_bytes": elem["out"]}
    if raw_abi is not None and raw_abi != calculated_abi:
        raise ValueError(f"raw case ABI {raw_abi} does not match active config {calculated_abi}")
    memory = {0: ("uop", elem["uop"]), 1: ("wgt", elem["wgt"]),
              2: ("inp", elem["inp"]), 3: ("acc", elem["acc"]),
              4: ("out", elem["out"]), 5: ("acc", elem["acc"] // 4)}
    output.mkdir(parents=True, exist_ok=True)
    insn_alloc = allocation_for(before, before["insn_phy_addr"])
    insn_offset = before["insn_phy_addr"] - insn_alloc["phy_addr"]
    insn_bytes = before["insn_count"] * 16
    original_insn = output / "insn.original.bin"
    original_insn.write_bytes((raw / insn_alloc["file"]).read_bytes()[insn_offset:insn_offset + insn_bytes])

    with tempfile.TemporaryDirectory(prefix="vta-case-abi-") as temporary:
        abi = Path(temporary) / "vta_case_abi"
        build_abi(repo, abi)
        decoded = subprocess.check_output([str(abi), "inspect", str(original_insn)], text=True)
        arenas: dict[str, bytearray] = {}
        placements: dict[tuple[str, int], int] = {}
        patches: list[tuple[int, int]] = []
        accesses = []
        after_by_phy = {item["phy_addr"]: item for item in after["allocations"]}
        expected_out = bytearray()
        expected_outputs = []
        records = []
        used_by_region: dict[tuple[str, int], int] = {}
        for line in decoded.splitlines():
            index, opcode, memory_type, dram_base, y_size, x_size, x_stride = map(int, line.split())
            if opcode not in (LOAD, STORE) or x_size == 0:
                continue
            if memory_type not in memory:
                raise ValueError(f"instruction {index}: unsupported memory type {memory_type}")
            name, elem_bytes = memory[memory_type]
            address = dram_base * elem_bytes
            region = allocation_for(before, address)
            key = (name, region["phy_addr"])
            region_offset = address - region["phy_addr"]
            span = ((y_size - 1) * x_stride + x_size) * elem_bytes
            used_by_region[key] = max(used_by_region.get(key, 0), region_offset + span)
            records.append((index, opcode, memory_type, name, elem_bytes,
                            address, region, region_offset, span))

        for _, _, _, name, elem_bytes, _, region, _, _ in records:
            key = (name, region["phy_addr"])
            if key in placements:
                continue
            arena = arenas.setdefault(name, bytearray())
            start = align(len(arena), max(64, elem_bytes))
            arena.extend(bytes(start - len(arena)))
            placements[key] = start
            region_data = (raw / region["file"]).read_bytes()
            arena.extend(region_data[:used_by_region[key]])

        for (index, opcode, memory_type, name, elem_bytes,
             address, region, region_offset, span) in records:
            key = (name, region["phy_addr"])
            relocated_byte = placements[key] + region_offset
            if relocated_byte % elem_bytes:
                raise ValueError(f"instruction {index}: unaligned {name} address")
            patches.append((index, relocated_byte // elem_bytes))
            accesses.append({"insn": index, "opcode": opcode, "memory": name,
                             "original_phy_addr": address, "offset": relocated_byte,
                             "size": span, "element_bytes": elem_bytes})
            if opcode == STORE and memory_type == 4:
                post = after_by_phy[region["phy_addr"]]
                post_data = (raw / post["file"]).read_bytes()
                expected_offset = len(expected_out)
                expected_out.extend(post_data[region_offset:region_offset + span])
                expected_outputs.append({"address": DEFAULT_BASES["out"] + relocated_byte,
                                         "size": span, "expected_offset": expected_offset})
        patch_file = Path(temporary) / "patches.txt"
        patch_file.write_text("".join(f"{i} {base}\n" for i, base in patches))
        subprocess.run([str(abi), "patch", str(original_insn), str(output / "insn.bin"),
                        str(patch_file)], check=True)

    for name, data in arenas.items():
        data.extend(bytes(align(len(data)) - len(data)))
        (output / f"{name}.bin").write_bytes(data)
    (output / "expected_out.bin").write_bytes(expected_out)
    original_insn.unlink()
    sections = {"insn": {"file": "insn.bin", "address": DEFAULT_BASES["insn"]}}
    for name in arenas:
        sections[name] = {"file": f"{name}.bin", "address": DEFAULT_BASES[name]}
    sections["expected_out"] = {"file": "expected_out.bin"}
    for section in sections.values():
        path = output / section["file"]
        section.update(size=path.stat().st_size, sha256=sha256(path), alignment=64)
    manifest = {
        "format": "vta-soc-case-v1", "source_format": before["format"],
        "insn_count": before["insn_count"], "vcr_base_mode": "per-memory-type",
        "config": config, "sections": sections, "accesses": accesses,
        "expected_outputs": expected_outputs,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def validate(case: Path) -> None:
    manifest = json.loads((case / "manifest.json").read_text())
    if manifest["format"] != "vta-soc-case-v1":
        raise ValueError("unsupported case format")
    ranges = []
    for name, section in manifest["sections"].items():
        path = case / section["file"]
        if path.stat().st_size != section["size"] or sha256(path) != section["sha256"]:
            raise ValueError(f"section {name} size/hash mismatch")
        if "address" in section:
            ranges.append((section["address"], section["address"] + section["size"], name))
    ranges.sort()
    for left, right in zip(ranges, ranges[1:]):
        if left[1] > right[0]:
            raise ValueError(f"sections {left[2]} and {right[2]} overlap")
    print(f"VTA case PASS: {manifest['insn_count']} instructions, {len(ranges)} loadable sections")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pack_parser = sub.add_parser("pack")
    pack_parser.add_argument("raw", type=Path)
    pack_parser.add_argument("output", type=Path)
    pack_parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("case", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.raw.resolve(), args.output.resolve(), args.repo.resolve())
        validate(args.output.resolve())
    else:
        validate(args.case.resolve())


if __name__ == "__main__":
    main()
