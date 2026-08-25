#!/usr/bin/env python3
"""Fail if Python or simulator libraries cross the TVM/VTA boundary."""

from __future__ import annotations

import os
from pathlib import Path

import tvm
import vta
from vta.libinfo import find_libvta


workspace = Path(__file__).resolve().parents[2]
vta_root = workspace / "tvm-vta"
tvm_root = workspace / "tvm"
vta_module = Path(vta.__file__).resolve()
tvm_module = Path(tvm.__file__).resolve()

assert vta_module.is_relative_to(vta_root / "python"), vta_module
assert tvm_module.is_relative_to(tvm_root / "python"), tvm_module
assert str(tvm_root / "vta" / "python") not in os.environ.get("PYTHONPATH", "")

backend = vta.get_env().TARGET
if backend in ("sim", "tsim"):
    library = Path(find_libvta("libvta_fsim" if backend == "sim" else "libvta_tsim")[0])
    assert library.is_relative_to(vta_root / "build"), library

for stale in (tvm_root / "build").glob("libvta_*"):
    raise AssertionError(f"stale TVM-owned VTA library exists: {stale}")

print(f"isolation PASS: tvm={tvm_module}, vta={vta_module}, backend={backend}")
