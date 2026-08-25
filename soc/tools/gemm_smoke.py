#!/usr/bin/env python3
"""Run a 1x16 by 16x16 GEMM through the full LiteX/VTA simulation."""

from __future__ import annotations

import argparse
import os
import pty
import select
import subprocess
import tempfile
import time
from pathlib import Path


SECTIONS = {
    "insn.bin": 0x40010000,
    "uop.bin": 0x40011000,
    "input.bin": 0x40012000,
    "weight.bin": 0x40013000,
    "bias.bin": 0x40014000,
    "output.bin": 0x40015000,
}
VCR = 0xF1000000
TRANSCRIPT = bytearray()


def read_until(fd: int, marker: bytes, timeout: float) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout
    while marker not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(bytes(data[-500:]))
        ready, _, _ = select.select([fd], [], [], remaining)
        if ready:
            data.extend(os.read(fd, 4096))
    return bytes(data)


def command(fd: int, text: str, timeout: float) -> bytes:
    os.write(fd, text.encode() + b"\r")
    output = read_until(fd, b"> ", timeout)
    TRANSCRIPT.extend(output)
    return output


def write_blob(fd: int, address: int, data: bytes, timeout: float) -> None:
    data += bytes((-len(data)) % 4)
    for offset in range(0, len(data), 4):
        value = int.from_bytes(data[offset:offset + 4], "little")
        command(fd, f"mem_write 0x{address + offset:08x} 0x{value:08x}", timeout)


def build_workload(soc_dir: Path, output: Path) -> None:
    defines = [
        "VTA_LOG_INP_WIDTH=3", "VTA_LOG_WGT_WIDTH=3", "VTA_LOG_ACC_WIDTH=5",
        "VTA_LOG_OUT_WIDTH=3", "VTA_LOG_BATCH=0", "VTA_LOG_BLOCK_IN=4",
        "VTA_LOG_BLOCK_OUT=4", "VTA_LOG_BUS_WIDTH=6", "VTA_LOG_UOP_BUFF_SIZE=15",
        "VTA_LOG_INP_BUFF_SIZE=15", "VTA_LOG_WGT_BUFF_SIZE=18",
        "VTA_LOG_ACC_BUFF_SIZE=17",
    ]
    binary = output / "gen_gemm_smoke"
    subprocess.run(
        ["/usr/bin/clang++", "-std=c++17", f"-I{soc_dir.parent / 'include'}",
         *(f"-D{x}" for x in defines), str(soc_dir / "tools/gen_gemm_smoke.cc"),
         "-o", str(binary)], check=True)
    subprocess.run([str(binary), str(output)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateware", type=Path, default=Path("build/sim/gateware"))
    parser.add_argument("--timeout", type=float, default=40.0)
    args = parser.parse_args()
    soc_dir = Path(__file__).resolve().parents[1]
    gateware = args.gateware.resolve()
    env = os.environ.copy()
    conda_env = Path(env.get("CONDA_ENV_DIR", "/Users/xdl/tools/miniforge3-26.1.1-3/envs/tvm_py310"))
    env.setdefault("DYLD_LIBRARY_PATH", str(conda_env / "lib"))

    with tempfile.TemporaryDirectory(prefix="vta-gemm-") as temporary:
        workload = Path(temporary)
        build_workload(soc_dir, workload)
        master, slave = pty.openpty()
        process = subprocess.Popen(
            [str(gateware / "obj_dir/Vsim"), "sim_config.js"], cwd=gateware,
            stdin=slave, stdout=slave, stderr=slave, close_fds=True, env=env)
        os.close(slave)
        try:
            read_until(master, b"> ", args.timeout)
            for name, address in SECTIONS.items():
                write_blob(master, address, (workload / name).read_bytes(), args.timeout)
            command(master, "flush_cpu_dcache", args.timeout)
            command(master, "flush_l2_cache", args.timeout)
            registers = [7, SECTIONS["insn.bin"], SECTIONS["uop.bin"],
                         SECTIONS["input.bin"], SECTIONS["weight.bin"],
                         SECTIONS["bias.bin"], SECTIONS["output.bin"]]
            for index, value in enumerate(registers, start=2):
                command(master, f"mem_write 0x{VCR + 4 * index:08x} 0x{value:08x}", args.timeout)
            programmed = command(master, f"mem_read 0x{VCR:08x} 0x28", args.timeout)
            command(master, f"mem_write 0x{VCR:08x} 0x00000001", args.timeout)
            status = b""
            for _ in range(100):
                status = command(master, f"mem_read 0x{VCR:08x} 0x8", args.timeout)
                if b"02 00 00 00" in status:
                    break
            result = command(master, f"mem_read 0x{SECTIONS['output.bin']:08x} 0x10", args.timeout)
            if b"02 00 00 00" not in status:
                axi = b"\n".join(
                    line for line in TRANSCRIPT.splitlines()
                    if b"VTA_AXI" in line or b"VTA_CTRL" in line)
                raise RuntimeError(
                    f"VTA did not finish; VCR={programmed!r}; AXI trace={axi[-8000:]!r}")
            if b"10 " * 15 + b"10" not in result:
                raise RuntimeError(f"GEMM result mismatch: {result!r}")
            print("VTA GEMM smoke PASS: 1x16 * 16x16 produced sixteen int8 values of 16")
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
