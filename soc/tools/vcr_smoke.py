#!/usr/bin/env python3
"""Exercise the CPU-to-VTA VCR path through the LiteX BIOS console."""

from __future__ import annotations

import argparse
import os
import pty
import select
import subprocess
import time
from pathlib import Path


def read_until(master: int, marker: bytes, timeout: float) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout
    while marker not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timeout waiting for {marker!r}; tail={bytes(data[-400:])!r}")
        ready, _, _ = select.select([master], [], [], remaining)
        if ready:
            data.extend(os.read(master, 4096))
    return bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateware", type=Path, default=Path("build/sim/gateware"))
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    gateware = args.gateware.resolve()
    env = os.environ.copy()
    conda_env = Path(env.get(
        "CONDA_ENV_DIR", "/Users/xdl/tools/miniforge3-26.1.1-3/envs/tvm_py310"))
    env.setdefault("DYLD_LIBRARY_PATH", str(conda_env / "lib"))
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [str(gateware / "obj_dir/Vsim"), "sim_config.js"],
        cwd=gateware,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
        env=env,
    )
    os.close(slave)
    try:
        read_until(master, b"litex", args.timeout)
        read_until(master, b"> ", args.timeout)
        os.write(master, b"mem_write 0xf1000008 0x12345678\r")
        read_until(master, b"> ", args.timeout)
        os.write(master, b"mem_read 0xf1000000 0x10\r")
        output = read_until(master, b"> ", args.timeout)
        if b"78 56 34 12" not in output:
            raise RuntimeError(f"VTA VCR readback mismatch: {output!r}")
        print("VTA VCR smoke PASS: reg[0x08] read back 0x12345678")
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
