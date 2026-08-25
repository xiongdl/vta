"""Out-of-tree integration workloads spanning GEMM, ACC and requantize ALU."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from vta.testing import simulator  # noqa: F401; loads the selected standalone runtime


TEST_ROOT = Path(__file__).resolve().parents[1]


def load_test_module(filename: str):
    path = TEST_ROOT / filename
    spec = importlib.util.spec_from_file_location(f"vta_out_of_tree_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dense_gemm_bias_clip_integration():
    dense = load_test_module("test_relay_dense_fsim.py")
    dense.test_relay_qnn_dense_fsim()


def test_conv2d_non_power_of_two_requantize_integration():
    conv2d = load_test_module("test_relay_conv2d_fsim.py")
    conv2d.test_relay_qnn_conv2d_non_power_of_two_right_shift_fsim()
