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
    if dense is None or requantize is None:
        raise ValueError("Expected qnn.dense followed by qnn.requantize")
    if not isinstance(dense.args[1], relay.Constant):
        raise ValueError("VTA dense weights must be constant")
    values = [_scalar(dense.args[i]) for i in range(2, 6)]
    values += [_scalar(requantize.args[i]) for i in range(1, 5)]
    if values != [0, 0, 1.0, 1.0, 1.0, 0, 1.0, 0]:
        raise ValueError("Only symmetric scale=1, zero-point=0 QNN dense is supported")
    return typed, dense, requantize


def compile_qnn_dense(function, name="vta_qnn_dense"):
    """Compile one supported QNN dense Relay function into a VTA module."""
    env = get_env()
    typed, dense, _ = _extract_dense(function)
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
    output = te.compute(output_shape, lambda *idx: accum(*idx).astype(env.out_dtype), name="output")

    schedule = te.create_schedule(output.op)
    schedule[data_buf].set_scope(env.inp_scope)
    schedule[weight_buf].set_scope(env.wgt_scope)
    schedule[accum].set_scope(env.acc_scope)
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

    lowered = lower(schedule, [data, weight, output], simple_mode=True)
    module = build(
        schedule,
        [data, weight, output],
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
    return DenseArtifact(module, lowered, packed_weight, (batch, in_features), (batch, out_features))


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
