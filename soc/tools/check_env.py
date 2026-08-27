#!/usr/bin/env python3
"""Check the isolated Conda/venv tools required by the SoC flow."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")


def main() -> None:
    soc = Path(__file__).resolve().parents[1]
    workspace = soc.parents[1]
    conda = Path(sys.base_prefix)
    for name in ("sbt", "verilator", "cmake", "make"):
        require_file(conda / "bin" / name, f"tvm_py310 tool {name}")
    require_file(conda / "lib" / "jvm", "tvm_py310 OpenJDK")
    require_file(
        workspace / "tools" / "xpack-riscv-none-elf-gcc-15.2.0-1" /
        "bin" / "riscv-none-elf-gcc",
        "RISC-V toolchain",
    )
    for module in ("litex", "litedram", "migen"):
        if importlib.util.find_spec(module) is None:
            raise SystemExit(f"missing Python package in soc/.venv: {module}")
    if platform.system() == "Darwin":
        if shutil.which("xcrun") is None:
            raise SystemExit("missing xcrun; install Apple Command Line Tools")
        sdk = subprocess.check_output(
            ("xcrun", "--show-sdk-path"), text=True).strip()
        require_file(Path(sdk) / "usr" / "include" / "c++" / "v1",
                     "Apple SDK libc++ headers")
    print(f"SoC environment valid: conda={conda}, venv={sys.prefix}")


if __name__ == "__main__":
    main()
