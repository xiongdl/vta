"""Unit tests for graphpack-free VTA Relay partitioning."""

import numpy as np
import tvm
from tvm import relay

import vta


def _qnn_conv_module(channels=16):
    data = relay.var("data", shape=(1, channels, 8, 8), dtype="int8")
    weight = relay.const(np.ones((channels, channels, 3, 3), dtype="int8"))
    conv = relay.qnn.op.conv2d(
        data,
        weight,
        relay.const(0, "int32"),
        relay.const(0, "int32"),
        relay.const(1.0, "float32"),
        relay.const(1.0, "float32"),
        padding=(1, 1),
        channels=channels,
        kernel_size=(3, 3),
        out_dtype="int32",
    )
    req = relay.qnn.op.requantize(
        conv,
        relay.const(1.0, "float32"),
        relay.const(0, "int32"),
        relay.const(1.0, "float32"),
        relay.const(0, "int32"),
        out_dtype="int8",
    )
    return tvm.IRModule.from_expr(relay.Function([data], req))


def test_partition_qnn_conv2d():
    partitioned = vta.partition_for_vta(_qnn_conv_module())
    compiler_functions = [
        func
        for func in partitioned.functions.values()
        if isinstance(func, relay.Function)
        and func.attrs
        and func.attrs.get("Compiler") == "vta"
    ]
    assert len(compiler_functions) == 1
    assert compiler_functions[0].attrs["Primitive"] == 1


def test_imports_out_of_tree_package():
    assert "/tvm-vta/python/vta/" in vta.__file__

