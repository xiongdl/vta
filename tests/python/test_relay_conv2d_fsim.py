"""Relay QNN conv2d composite -> VTA TIR -> standalone FSIM."""

import numpy as np
import tvm
from tvm import relay, rpc
from tvm.contrib import utils
from tvm.topi.testing import conv2d_nchw_python

import vta
from vta.testing import simulator  # pylint: disable=unused-import


def _run_conv2d(channels, out_channels, size, kernel_size, stride, padding,
                requant_input_scale=0.125, requant_output_scale=0.125):
    env = vta.get_env()
    batch = env.BATCH
    weight = np.random.randint(
        -3, 4, (out_channels, channels, kernel_size, kernel_size)
    ).astype("int8")
    bias = np.random.randint(-5, 6, (out_channels,)).astype("int32")
    data_var = relay.var("data", shape=(batch, channels, size, size), dtype="int8")
    conv = relay.qnn.op.conv2d(
        data_var,
        relay.const(weight),
        relay.const(0, "int32"),
        relay.const(0, "int32"),
        relay.const(0.5, "float32"),
        relay.const(0.25, "float32"),
        strides=(stride, stride),
        padding=(padding, padding),
        channels=out_channels,
        kernel_size=(kernel_size, kernel_size),
        out_dtype="int32",
    )
    biased = relay.nn.bias_add(conv, relay.const(bias))
    requantized = relay.qnn.op.requantize(
        biased,
        relay.const(requant_input_scale, "float32"),
        relay.const(4, "int32"),
        relay.const(requant_output_scale, "float32"),
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
    padded_channels = artifact.packed_input_shape[1] * env.BLOCK_IN
    padded_data = np.zeros((batch, padded_channels, size, size), dtype="int8")
    padded_data[:, :channels] = data
    packed_data = padded_data.reshape(
        batch // env.BATCH,
        env.BATCH,
        padded_channels // env.BLOCK_IN,
        env.BLOCK_IN,
        size,
        size,
    ).transpose(0, 2, 4, 5, 1, 3)
    output = tvm.nd.empty(artifact.packed_output_shape, env.out_dtype, device)
    module(
        tvm.nd.array(packed_data, device),
        tvm.nd.array(artifact.packed_weight, device),
        tvm.nd.array(artifact.packed_bias, device),
        output,
    )
    packed_actual = output.numpy().transpose(0, 4, 1, 5, 2, 3)
    physical_shape = (
        batch,
        artifact.packed_output_shape[1] * env.BLOCK_OUT,
        artifact.output_shape[2],
        artifact.output_shape[3],
    )
    actual = packed_actual.reshape(physical_shape)[:, :out_channels]
    expected = conv2d_nchw_python(
        data.astype("int32"), weight.astype("int32"), stride, padding
    )
    multiplier, shift = vta.quantize_multiplier(requant_input_scale / requant_output_scale)
    requantize = np.vectorize(
        lambda value: vta.cmsis_nn_requantize(value - 4, multiplier, shift), otypes=[np.int32]
    )
    expected = requantize(expected + bias.reshape(1, -1, 1, 1)) - 2
    expected = np.clip(expected, -40, 39).astype("int8")
    np.testing.assert_equal(actual, expected)


def test_relay_qnn_conv2d_fsim():
    env = vta.get_env()
    _run_conv2d(env.BLOCK_IN, env.BLOCK_OUT, 5, 3, 1, 1)


def test_relay_qnn_conv2d_stride2_unaligned_channels():
    _run_conv2d(3, 5, 7, 3, 2, 0)


def test_relay_qnn_conv2d_non_power_of_two_requantize_fsim():
    env = vta.get_env()
    _run_conv2d(env.BLOCK_IN, env.BLOCK_OUT, 5, 3, 1, 1, 0.125, 0.2)


def test_single_primfunc_conv_residual_acc8():
    """One PrimFunc containing GEMM, ACC_8BIT shortcut load and residual ALU."""
    env = vta.get_env()
    shape = (env.BATCH, env.BLOCK_IN, 1, 1)
    data_var = relay.var("data", shape=shape, dtype="int8")
    weight = np.zeros((env.BLOCK_OUT, env.BLOCK_IN, 1, 1), dtype="int8")
    for channel in range(env.BLOCK_IN):
        weight[channel, channel, 0, 0] = 1
    conv = relay.qnn.op.conv2d(
        data_var, relay.const(weight), relay.const(0, "int32"), relay.const(0, "int32"),
        relay.const(1.0, "float32"), relay.const(1.0, "float32"), channels=env.BLOCK_OUT,
        kernel_size=(1, 1), out_dtype="int32",
    )
    output = relay.qnn.op.requantize(
        conv, relay.const(1.0, "float32"), relay.const(0, "int32"),
        relay.const(1.0, "float32"), relay.const(0, "int32"), out_dtype="int8",
    )
    function = relay.Function([data_var], output)
    from vta.compiler.conv2d import compile_qnn_conv2d
    artifact = compile_qnn_conv2d(function, name="single_primfunc_residual", residual=True)
    temp = utils.tempdir()
    artifact.module.save(temp.relpath("single_primfunc_residual.o"))
    remote = rpc.LocalSession()
    remote.upload(temp.relpath("single_primfunc_residual.o"))
    module = remote.load_module("single_primfunc_residual.o")
    device = remote.ext_dev(0)
    data = np.random.randint(-16, 17, shape).astype("int8")
    packed = data.reshape(1, env.BATCH, 1, env.BLOCK_IN, 1, 1).transpose(0, 2, 4, 5, 1, 3)
    result = tvm.nd.empty(artifact.packed_output_shape, "int8", device)
    module(
        tvm.nd.array(packed, device), tvm.nd.array(artifact.packed_weight, device),
        tvm.nd.array(artifact.packed_bias, device), tvm.nd.array(packed, device), result,
    )
    np.testing.assert_equal(result.numpy(), (packed + packed).astype("int8"))
