"""Minimal graphpack-free QNN dense compiler for VTA."""

from dataclasses import dataclass

import numpy as np
import tvm
from tvm import relay, te

from ..environment import get_env
from ..build_module import build, lower


@dataclass
class DenseArtifact:
    module: tvm.runtime.Module
    lowered: tvm.IRModule
    packed_weight: np.ndarray
    packed_bias: np.ndarray
    input_shape: tuple
    output_shape: tuple


def _find_call(expr, name):
    if isinstance(expr, relay.Call):
        if isinstance(expr.op, tvm.ir.Op) and expr.op.name == name:
            return expr
        if isinstance(expr.op, relay.Function):
            found = _find_call(expr.op.body, name)
            if found is not None:
                return found
        for arg in expr.args:
            found = _find_call(arg, name)
            if found is not None:
                return found
    return None


def _scalar(constant):
    if not isinstance(constant, relay.Constant):
        raise ValueError("VTA QNN dense currently requires constant quantization parameters")
    return constant.data.numpy().item()


def _extract_dense(function):
    module = tvm.IRModule({"main": function.without_attr("global_symbol")})
    typed = relay.transform.InferType()(module)["main"]
    dense = _find_call(typed.body, "qnn.dense")
    requantize = _find_call(typed.body, "qnn.requantize")
    bias_add = _find_call(typed.body, "nn.bias_add")
    clip = _find_call(typed.body, "clip")
    if dense is None or requantize is None:
        raise ValueError("Expected qnn.dense followed by qnn.requantize")
    if not isinstance(dense.args[1], relay.Constant):
        raise ValueError("VTA dense weights must be constant")
    input_zero_point, kernel_zero_point = (_scalar(dense.args[i]) for i in range(2, 4))
    input_scale, kernel_scale = (_scalar(dense.args[i]) for i in range(4, 6))
    requant_input_scale = _scalar(requantize.args[1])
    requant_input_zero_point = _scalar(requantize.args[2])
    requant_output_scale = _scalar(requantize.args[3])
    requant_output_zero_point = _scalar(requantize.args[4])
    if input_zero_point != 0 or kernel_zero_point != 0:
        raise ValueError("VTA dense currently requires zero input and kernel zero-points")
    if not np.isclose(input_scale * kernel_scale, requant_input_scale):
        raise ValueError("qnn.dense scales must match the requantize input scale")
    if not np.isclose(requant_input_scale, requant_output_scale):
        raise ValueError("VTA ALU cannot exactly express requantize with unequal scales")
    if requantize.checked_type.dtype != "int8":
        raise ValueError("VTA dense currently requires int8 requantize output")
    bias = None
    if bias_add is not None:
        if not isinstance(bias_add.args[1], relay.Constant):
            raise ValueError("VTA dense bias must be constant")
        bias = bias_add.args[1].data.numpy()
    clip_bounds = (-128, 127)
    if clip is not None:
        clip_bounds = (
            max(clip_bounds[0], int(clip.attrs.a_min)),
            min(clip_bounds[1], int(clip.attrs.a_max)),
        )
    zero_point_correction = requant_output_zero_point - requant_input_zero_point
    return typed, dense, bias, clip_bounds, zero_point_correction


def compile_qnn_dense(function, name="vta_qnn_dense"):
    """Compile one supported QNN dense Relay function into a VTA module."""
    env = get_env()
    typed, dense, raw_bias, clip_bounds, zero_point_correction = _extract_dense(function)
    batch, in_features = (int(x) for x in dense.args[0].checked_type.shape)
    out_features, weight_in = (int(x) for x in dense.args[1].checked_type.shape)
    if weight_in != in_features:
        raise ValueError("Dense input/weight dimensions do not match")
    if batch % env.BATCH or in_features % env.BLOCK_IN or out_features % env.BLOCK_OUT:
        raise ValueError("Dense dimensions must align to VTA BATCH/BLOCK_IN/BLOCK_OUT")

    data_shape = (batch // env.BATCH, in_features // env.BLOCK_IN, env.BATCH, env.BLOCK_IN)
    weight_shape = (
        out_features // env.BLOCK_OUT,
        in_features // env.BLOCK_IN,
        env.BLOCK_OUT,
        env.BLOCK_IN,
    )
    output_shape = (batch // env.BATCH, out_features // env.BLOCK_OUT, env.BATCH, env.BLOCK_OUT)
    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    weight = te.placeholder(weight_shape, name="weight", dtype=env.wgt_dtype)
    bias = te.placeholder(output_shape, name="bias", dtype=env.acc_dtype)
    data_buf = te.compute(data_shape, lambda *idx: data(*idx), name="data_buf")
    weight_buf = te.compute(weight_shape, lambda *idx: weight(*idx), name="weight_buf")
    ko = te.reduce_axis((0, in_features // env.BLOCK_IN), name="ko")
    ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")
    accum = te.compute(
        output_shape,
        lambda bo, co, bi, ci: te.sum(
            data_buf[bo, ko, bi, ki].astype(env.acc_dtype)
            * weight_buf[co, ko, ci, ki].astype(env.acc_dtype),
            axis=[ko, ki],
        ),
        name="accum",
    )
    biased = te.compute(output_shape, lambda *idx: accum(*idx) + bias(*idx), name="biased")
    clip_min, clip_max = clip_bounds
    clipped_min = te.compute(
        output_shape,
        lambda *idx: tvm.te.min(biased(*idx), tvm.tir.const(clip_max, env.acc_dtype)),
        name="clip_max",
    )
    narrowed = te.compute(
        output_shape,
        lambda *idx: tvm.te.max(clipped_min(*idx), tvm.tir.const(clip_min, env.acc_dtype)),
        name="clip_min",
    )
    output = te.compute(output_shape, lambda *idx: narrowed(*idx).astype(env.out_dtype), name="output")

    schedule = te.create_schedule(output.op)
    schedule[data_buf].set_scope(env.inp_scope)
    schedule[weight_buf].set_scope(env.wgt_scope)
    schedule[accum].set_scope(env.acc_scope)
    schedule[biased].set_scope(env.acc_scope)
    schedule[biased].pragma(schedule[biased].op.axis[0], env.alu)
    for stage in (clipped_min, narrowed):
        schedule[stage].set_scope(env.acc_scope)
        schedule[stage].pragma(schedule[stage].op.axis[0], env.alu)
    bias_buf = schedule.cache_read(bias, env.acc_scope, [biased])
    schedule[bias_buf].pragma(schedule[bias_buf].op.axis[0], env.dma_copy)
    if in_features // env.BLOCK_IN > 1:
        schedule[data_buf].compute_at(schedule[accum], ko)
        schedule[weight_buf].compute_at(schedule[accum], ko)
    schedule[data_buf].pragma(schedule[data_buf].op.axis[0], env.dma_copy)
    schedule[weight_buf].pragma(schedule[weight_buf].op.axis[0], env.dma_copy)
    schedule[accum].reorder(
        ko,
        schedule[accum].op.axis[0],
        schedule[accum].op.axis[1],
        schedule[accum].op.axis[2],
        schedule[accum].op.axis[3],
        ki,
    )
    schedule[accum].tensorize(schedule[accum].op.axis[2], env.gemm)
    schedule[output].pragma(schedule[output].op.axis[0], env.dma_copy)

    args = [data, weight, bias, output]
    lowered = lower(schedule, args, simple_mode=True)
    module = build(
        schedule,
        args,
        tvm.target.Target("ext_dev", host=env.target_host),
        name=name,
    )
    raw_weight = dense.args[1].data.numpy().astype(env.wgt_dtype)
    packed_weight = raw_weight.reshape(
        out_features // env.BLOCK_OUT,
        env.BLOCK_OUT,
        in_features // env.BLOCK_IN,
        env.BLOCK_IN,
    ).transpose(0, 2, 1, 3)
    if raw_bias is None:
        raw_bias = np.zeros((out_features,), dtype=env.acc_dtype)
    raw_bias = raw_bias.astype(env.acc_dtype) + np.asarray(zero_point_correction, env.acc_dtype)
    packed_bias = np.broadcast_to(raw_bias.reshape(1, -1), (batch, out_features)).copy()
    packed_bias = packed_bias.reshape(
        batch // env.BATCH, env.BATCH, out_features // env.BLOCK_OUT, env.BLOCK_OUT
    ).transpose(0, 2, 1, 3).astype(env.acc_dtype)
    return DenseArtifact(
        module, lowered, packed_weight, packed_bias, (batch, in_features), (batch, out_features)
    )


def compile_partitioned_dense(mod):
    """Compile the single VTA QNN dense region in a partitioned Relay module."""
    candidates = []
    for global_var, function in mod.functions.items():
        if (
            isinstance(function, relay.Function)
            and function.attrs
            and function.attrs.get("Compiler") == "vta"
            and _find_call(function.body, "qnn.dense") is not None
        ):
            candidates.append((global_var.name_hint, function))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one VTA dense region, found {len(candidates)}")
    name, function = candidates[0]
    return compile_qnn_dense(function, name=name)
