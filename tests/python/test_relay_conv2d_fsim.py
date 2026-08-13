"""Relay QNN conv2d composite -> VTA TIR -> standalone FSIM."""

import numpy as np
import tvm
from tvm import relay, rpc
from tvm.contrib import utils
from tvm.topi.testing import conv2d_nchw_python

import vta


def test_relay_qnn_conv2d_fsim():
    env = vta.get_env()
    batch, channels, size = env.BATCH, env.BLOCK_IN, 5
    weight = np.random.randint(-3, 4, (env.BLOCK_OUT, channels, 3, 3)).astype("int8")
    bias = np.random.randint(-5, 6, (env.BLOCK_OUT,)).astype("int32")
    data_var = relay.var("data", shape=(batch, channels, size, size), dtype="int8")
    conv = relay.qnn.op.conv2d(
        data_var,
        relay.const(weight),
        relay.const(0, "int32"),
        relay.const(0, "int32"),
        relay.const(0.5, "float32"),
        relay.const(0.25, "float32"),
        padding=(1, 1),
        channels=env.BLOCK_OUT,
        kernel_size=(3, 3),
        out_dtype="int32",
    )
    biased = relay.nn.bias_add(conv, relay.const(bias))
    requantized = relay.qnn.op.requantize(
        biased,
        relay.const(0.125, "float32"),
        relay.const(4, "int32"),
        relay.const(0.125, "float32"),
        relay.const(-2, "int32"),
        out_dtype="int8",
    )
    function = relay.Function([data_var], relay.clip(requantized, a_min=-40, a_max=39))
    artifact = vta.compile(tvm.IRModule.from_expr(function))
    assert "VTAPushGEMMOp" in str(artifact.lowered)

    temp = utils.tempdir()
    artifact.module.save(temp.relpath("conv2d.o"))
    remote = rpc.LocalSession()
    remote.upload(temp.relpath("conv2d.o"))
    module = remote.load_module("conv2d.o")
    device = remote.ext_dev(0)
    data = np.random.randint(-3, 4, (batch, channels, size, size)).astype("int8")
    packed_data = data.reshape(
        batch // env.BATCH,
        env.BATCH,
        channels // env.BLOCK_IN,
        env.BLOCK_IN,
        size,
        size,
    ).transpose(0, 2, 4, 5, 1, 3)
    packed_output_shape = (
        batch // env.BATCH, env.BLOCK_OUT // env.BLOCK_OUT, size, size, env.BATCH, env.BLOCK_OUT
    )
    output = tvm.nd.empty(packed_output_shape, env.out_dtype, device)
    module(
        tvm.nd.array(packed_data, device),
        tvm.nd.array(artifact.packed_weight, device),
        tvm.nd.array(artifact.packed_bias, device),
        output,
    )
    actual = output.numpy().transpose(0, 4, 1, 5, 2, 3).reshape(artifact.output_shape)
    expected = conv2d_nchw_python(data.astype("int32"), weight.astype("int32"), 1, 1)
    expected = np.clip(expected + bias.reshape(1, -1, 1, 1) - 4 - 2, -40, 39).astype("int8")
    np.testing.assert_equal(actual, expected)
