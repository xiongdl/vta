"""Regression checks for the TVM/VTA repository boundary."""

from pathlib import Path

import tvm
import vta
from vta.libinfo import find_libvta


WORKSPACE = Path(__file__).resolve().parents[3]


def test_python_modules_are_out_of_tree():
    assert Path(tvm.__file__).resolve().is_relative_to(WORKSPACE / "tvm" / "python")
    assert Path(vta.__file__).resolve().is_relative_to(WORKSPACE / "tvm-vta" / "python")


def test_runtime_library_is_owned_by_tvm_vta():
    name = "libvta_tsim" if vta.get_env().TARGET == "tsim" else "libvta_fsim"
    library = Path(find_libvta(name)[0]).resolve()
    assert library.is_relative_to(WORKSPACE / "tvm-vta" / "build")
