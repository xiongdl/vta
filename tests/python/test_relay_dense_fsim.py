"""Relay QNN dense composite -> VTA TIR -> standalone FSIM."""

import numpy as np
import tvm
from tvm import relay, rpc
from tvm.contrib import utils

import vta


def _make_dense(
    batch,
    in_features,
    out_features,
    weight,
    bias=None,
    clip=None,
    input_scale=1.0,
    kernel_scale=1.0,
    requant_input_zero_point=0,
    output_zero_point=0,
    output_scale=None,
):
    requant_scale = input_scale * kernel_scale
    if output_scale is None:
        output_scale = requant_scale
    data = relay.var("data", shape=(batch, in_features), dtype="int8")
    dense = relay.qnn.op.dense(
        data,
        relay.const(weight),
        relay.const(0, "int32"),
        relay.const(0, "int32"),
        relay.const(input_scale, "float32"),
        relay.const(kernel_scale, "float32"),
        units=out_features,
        out_dtype="int32",
    )
    value = dense if bias is None else relay.nn.bias_add(dense, relay.const(bias))
    output = relay.qnn.op.requantize(
        value,
        relay.const(requant_scale, "float32"),
        relay.const(requant_input_zero_point, "int32"),
        relay.const(output_scale, "float32"),
        relay.const(output_zero_point, "int32"),
        out_dtype="int8",
    )
    if clip is not None:
        output = relay.clip(output, a_min=clip[0], a_max=clip[1])
    return relay.Function([data], output)


def test_relay_qnn_dense_fsim():
    env = vta.get_env()
    batch, in_features, out_features = env.BATCH, 2 * env.BLOCK_IN, env.BLOCK_OUT
    weight = np.random.randint(-4, 5, (out_features, in_features)).astype("int8")
    bias = np.random.randint(-8, 9, (out_features,)).astype("int32")
    function = _make_dense(
        batch,
        in_features,
        out_features,
        weight,
        bias=bias,
        clip=(-32, 31),
        input_scale=0.5,
        kernel_scale=0.25,
        requant_input_zero_point=7,
        output_zero_point=-3,
    )
    artifact = vta.compile(tvm.IRModule.from_expr(function))
    assert "VTAPushGEMMOp" in str(artifact.lowered)

    temp = utils.tempdir()
    artifact.module.save(temp.relpath("dense.o"))
    remote = rpc.LocalSession()
    remote.upload(temp.relpath("dense.o"))
    module = remote.load_module("dense.o")
    device = remote.ext_dev(0)

    data = np.random.randint(-4, 5, (batch, in_features)).astype("int8")
    packed_data = data.reshape(
        batch // env.BATCH, env.BATCH, in_features // env.BLOCK_IN, env.BLOCK_IN
    ).transpose(0, 2, 1, 3)
    output_shape = (
        batch // env.BATCH,
        out_features // env.BLOCK_OUT,
        env.BATCH,
        env.BLOCK_OUT,
    )
    data_nd = tvm.nd.array(packed_data, device)
    weight_nd = tvm.nd.array(artifact.packed_weight, device)
    bias_nd = tvm.nd.array(artifact.packed_bias, device)
    output_nd = tvm.nd.empty(output_shape, env.out_dtype, device)
    module(data_nd, weight_nd, bias_nd, output_nd)

    expected = np.dot(data.astype("int32"), weight.T.astype("int32")) + bias - 7 - 3
    expected = np.clip(expected, -32, 31).astype("int8")
    np.testing.assert_equal(output_nd.numpy().reshape(batch, out_features), expected)


def test_relay_qnn_dense_rejects_inexact_requantize():
    env = vta.get_env()
    weight = np.zeros((env.BLOCK_OUT, env.BLOCK_IN), dtype="int8")
    function = _make_dense(
        env.BATCH,
        env.BLOCK_IN,
        env.BLOCK_OUT,
        weight,
        output_scale=0.5,
    )
    partitioned = vta.partition_for_vta(tvm.IRModule.from_expr(function))
    assert not any(
        isinstance(candidate, relay.Function)
        and candidate.attrs
        and candidate.attrs.get("Compiler") == "vta"
        for candidate in partitioned.functions.values()
    )
