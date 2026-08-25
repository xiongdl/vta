#!/usr/bin/env python3
"""Replay a relocatable VTA case through the LiteX simulation console."""

from __future__ import annotations

import argparse
import json
import os
import pty
import select
import subprocess
import time
from pathlib import Path


VCR = 0xF1000000
VCR_POINTERS = {"insn": 3, "uop": 4, "inp": 5, "wgt": 6, "acc": 7, "out": 8}


def read_until(fd: int, marker: bytes, timeout: float) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout
    while marker not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(bytes(data[-500:]))
        ready, _, _ = select.select([fd], [], [], remaining)
        if ready:
            chunk = os.read(fd, 4096)
            if not chunk:
                raise RuntimeError(f"simulation console closed before {marker!r}: {bytes(data[-500:])!r}")
            data.extend(chunk)
    return bytes(data)


def command(fd: int, text: str, timeout: float) -> bytes:
    os.write(fd, text.encode() + b"\r")
    return read_until(fd, b"> ", timeout)


def write_blob(fd: int, address: int, data: bytes, timeout: float) -> None:
    data += bytes((-len(data)) % 4)
    for offset in range(0, len(data), 4):
        value = int.from_bytes(data[offset:offset + 4], "little")
        command(fd, f"mem_write 0x{address + offset:08x} 0x{value:08x}", timeout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--gateware", type=Path, default=Path("build/sim/gateware"))
    parser.add_argument("--timeout", type=float, default=40.0)
    args = parser.parse_args()
    case = args.case.resolve()
    manifest = json.loads((case / "manifest.json").read_text())
    if manifest["format"] != "vta-soc-case-v1":
        raise ValueError("unsupported case format")
    env = os.environ.copy()
    conda_env = Path(env.get("CONDA_ENV_DIR", "/Users/xdl/tools/miniforge3-26.1.1-3/envs/tvm_py310"))
    env.setdefault("DYLD_LIBRARY_PATH", str(conda_env / "lib"))
    gateware = args.gateware.resolve()
    master, slave = pty.openpty()
    process = subprocess.Popen([str(gateware / "obj_dir/Vsim"), "sim_config.js"],
                               cwd=gateware, stdin=slave, stdout=slave, stderr=slave,
                               close_fds=True, env=env)
    os.close(slave)
    try:
        read_until(master, b"> ", args.timeout)
        for name, section in manifest["sections"].items():
            if "address" in section:
                write_blob(master, section["address"], (case / section["file"]).read_bytes(), args.timeout)
        command(master, "flush_cpu_dcache", args.timeout)
        command(master, "flush_l2_cache", args.timeout)
        command(master, f"mem_write 0x{VCR + 8:08x} 0x{manifest['insn_count']:08x}", args.timeout)
        for name, register in VCR_POINTERS.items():
            address = manifest["sections"].get(name, {}).get("address", 0)
            command(master, f"mem_write 0x{VCR + 4 * register:08x} 0x{address:08x}", args.timeout)
        command(master, f"mem_write 0x{VCR:08x} 0x00000001", args.timeout)
        status = b""
        for _ in range(100):
            status = command(master, f"mem_read 0x{VCR:08x} 0x8", args.timeout)
            if b"02 00 00 00" in status:
                break
        if b"02 00 00 00" not in status:
            raise RuntimeError(f"VTA case did not finish: {status[-500:]!r}")
        expected = (case / manifest["sections"]["expected_out"]["file"]).read_bytes()
        for check in manifest["expected_outputs"]:
            for offset in range(0, check["size"], 16):
                size = min(16, check["size"] - offset)
                actual = command(master, f"mem_read 0x{check['address'] + offset:08x} 0x{size:x}", args.timeout)
                begin = check["expected_offset"] + offset
                wanted = expected[begin:begin + size].hex(" ").encode()
                if wanted not in actual.lower():
                    raise RuntimeError(f"output mismatch at 0x{check['address'] + offset:x}: {actual!r}")
        print(f"VTA SoC case PASS: {manifest['insn_count']} instructions")
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        os.close(master)


if __name__ == "__main__":
    main()
