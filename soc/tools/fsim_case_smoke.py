#!/usr/bin/env python3
"""Run the minimal GEMM directly on FSIM and emit a raw replay case."""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

from vta_case import build_abi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["VTA_CASE_DUMP_DIR"] = str(output)
    soc = Path(__file__).resolve().parents[1]
    repo = soc.parent
    ctypes.CDLL(str(repo.parent / "tvm/build/libtvm.dylib"), ctypes.RTLD_GLOBAL)
    library = ctypes.CDLL(str(args.library.resolve()), ctypes.RTLD_GLOBAL)
    library.VTAMemAlloc.argtypes = [ctypes.c_size_t, ctypes.c_int]
    library.VTAMemAlloc.restype = ctypes.c_void_p
    library.VTAMemGetPhyAddr.argtypes = [ctypes.c_void_p]
    library.VTAMemGetPhyAddr.restype = ctypes.c_uint32
    library.VTAMemCopyFromHost.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.VTAMemCopyToHost.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.VTADeviceAlloc.restype = ctypes.c_void_p
    library.VTADeviceRun.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    library.VTAExportCase.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32]

    with tempfile.TemporaryDirectory(prefix="vta-fsim-case-") as temporary:
        work = Path(temporary)
        generator = work / "gen_gemm_smoke"
        abi = work / "vta_case_abi"
        subprocess.run(["/usr/bin/clang++", "-std=c++17", f"-I{repo / 'include'}",
                        "-DVTA_LOG_INP_WIDTH=3", "-DVTA_LOG_WGT_WIDTH=3",
                        "-DVTA_LOG_ACC_WIDTH=5", "-DVTA_LOG_OUT_WIDTH=3",
                        "-DVTA_LOG_BATCH=0", "-DVTA_LOG_BLOCK_IN=4",
                        "-DVTA_LOG_BLOCK_OUT=4", "-DVTA_LOG_BUS_WIDTH=6",
                        "-DVTA_LOG_UOP_BUFF_SIZE=15", "-DVTA_LOG_INP_BUFF_SIZE=15",
                        "-DVTA_LOG_WGT_BUFF_SIZE=18", "-DVTA_LOG_ACC_BUFF_SIZE=17",
                        str(soc / "tools/gen_gemm_smoke.cc"), "-o", str(generator)], check=True)
        subprocess.run([str(generator), str(work)], check=True)
        build_abi(repo, abi)
        names = ["insn", "uop", "input", "weight", "bias", "output"]
        buffers = {}
        physical = {}
        for name in names:
            data = (work / f"{name}.bin").read_bytes()
            pointer = library.VTAMemAlloc(len(data), 1)
            source = ctypes.create_string_buffer(data)
            library.VTAMemCopyFromHost(pointer, source, len(data))
            buffers[name] = (pointer, len(data))
            physical[name] = library.VTAMemGetPhyAddr(pointer)
        patch_file = work / "patches.txt"
        patch_file.write_text(
            f"0 {physical['uop'] // 4}\n1 {physical['bias'] // 64}\n"
            f"2 {physical['weight'] // 256}\n3 {physical['input'] // 16}\n"
            f"5 {physical['output'] // 16}\n")
        patched = work / "insn.patched.bin"
        subprocess.run([str(abi), "patch", str(work / "insn.bin"), str(patched),
                        str(patch_file)], check=True)
        insn_data = patched.read_bytes()
        insn_source = ctypes.create_string_buffer(insn_data)
        library.VTAMemCopyFromHost(buffers["insn"][0], insn_source, len(insn_data))
        device = library.VTADeviceAlloc()
        library.VTAExportCase(str(output).encode(), b"before", physical["insn"], 7)
        status = library.VTADeviceRun(device, physical["insn"], 7, 100000)
        library.VTAExportCase(str(output).encode(), b"after", physical["insn"], 7)
        result = ctypes.create_string_buffer(16)
        library.VTAMemCopyToHost(result, buffers["output"][0], 16)
        if status != 0 or result.raw != bytes([16]) * 16:
            raise RuntimeError(f"FSIM GEMM failed: status={status}, output={result.raw.hex()}")
    print(f"FSIM raw case PASS: {output}")


if __name__ == "__main__":
    main()
