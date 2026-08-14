"""Graphpack-free QNN conv2d compiler for VTA."""

from dataclasses import dataclass

import numpy as np
import tvm
from tvm import relay, te, topi

from ..build_module import build, lower
from ..environment import get_env
from .dense import _find_call, _scalar
from ..relay.quantization import quantize_multiplier


@dataclass
class Conv2DArtifact:
    module: tvm.runtime.Module
    lowered: tvm.IRModule
    packed_weight: np.ndarray
    packed_bias: np.ndarray
    input_shape: tuple
    output_shape: tuple
    packed_input_shape: tuple
    packed_output_shape: tuple


def _pair(value, name):
    values = tuple(int(x) for x in value)
    if len(values) != 2:
        raise ValueError(f"VTA conv2d requires two-dimensional {name}")
    return values


def _extract_conv2d(function):
    module = tvm.IRModule({"main": function.without_attr("global_symbol")})
    typed = relay.transform.InferType()(module)["main"]
    conv = _find_call(typed.body, "qnn.conv2d")
    requantize = _find_call(typed.body, "qnn.requantize")
    bias_add = _find_call(typed.body, "nn.bias_add")
    clip = _find_call(typed.body, "clip")
    if conv is None or requantize is None:
        raise ValueError("Expected qnn.conv2d followed by qnn.requantize")
    if not isinstance(conv.args[1], relay.Constant):
        raise ValueError("VTA conv2d weights must be constant")
    if conv.attrs.data_layout != "NCHW" or conv.attrs.kernel_layout != "OIHW":
        raise ValueError("VTA conv2d currently requires NCHW data and OIHW kernels")
    if _pair(conv.attrs.dilation, "dilation") != (1, 1):
        raise ValueError("VTA conv2d currently requires dilation=(1, 1)")

    input_zero_point, kernel_zero_point = (_scalar(conv.args[i]) for i in range(2, 4))
    input_scale = np.asarray(conv.args[4].data.numpy())
    kernel_scale = np.asarray(conv.args[5].data.numpy())
    requant_input_scale = np.asarray(requantize.args[1].data.numpy())
    requant_input_zero_point = _scalar(requantize.args[2])
    requant_output_scale = np.asarray(requantize.args[3].data.numpy())
    requant_output_zero_point = _scalar(requantize.args[4])
    if input_zero_point != 0 or kernel_zero_point != 0:
        raise ValueError("VTA conv2d currently requires zero input and kernel zero-points")
    if not np.allclose(input_scale * kernel_scale, requant_input_scale):
        raise ValueError("qnn.conv2d scales must match the requantize input scale")
    requant_params = [quantize_multiplier(value) for value in
                      np.asarray(requant_input_scale / requant_output_scale).reshape(-1)]
    if len(set(requant_params)) != 1:
        raise ValueError("VTA conv2d currently requires uniform requantization parameters")
    requant_multiplier, requant_shift = requant_params[0]
    if requantize.checked_type.dtype != "int8":
        raise ValueError("VTA conv2d currently requires int8 requantize output")

    bias = None
    if bias_add is not None:
        if not isinstance(bias_add.args[1], relay.Constant):
            raise ValueError("VTA conv2d bias must be constant")
        bias = bias_add.args[1].data.numpy()
    clip_bounds = (-128, 127)
    if clip is not None:
        clip_bounds = (
            max(clip_bounds[0], int(clip.attrs.a_min)),
            min(clip_bounds[1], int(clip.attrs.a_max)),
        )
    correction = -requant_input_zero_point
    return (typed, conv, bias, clip_bounds, correction, requant_multiplier,
            requant_shift, requant_output_zero_point)


def compile_qnn_conv2d(function, name="vta_qnn_conv2d", residual=False):
    """Compile one supported QNN conv2d Relay function into a VTA module."""
    env = get_env()
    (_, conv, raw_bias, clip_bounds, correction, requant_multiplier,
     requant_shift, output_zero_point) = _extract_conv2d(function)
    batch, in_channels, height, width = (int(x) for x in conv.args[0].checked_type.shape)
    out_channels, weight_in_per_group, kernel_h, kernel_w = (
        int(x) for x in conv.args[1].checked_type.shape
    )
    groups = int(conv.attrs.groups)
    if groups <= 0 or in_channels % groups or out_channels % groups:
        raise ValueError("Conv2d groups must divide input and output channels")
    if weight_in_per_group * groups != in_channels:
        raise ValueError("Conv2d input/kernel channel dimensions do not match")
    if batch % env.BATCH:
        raise ValueError("Conv2d batch must align to VTA BATCH")
    padded_in_channels = ((in_channels + env.BLOCK_IN - 1) // env.BLOCK_IN) * env.BLOCK_IN
    padded_out_channels = ((out_channels + env.BLOCK_OUT - 1) // env.BLOCK_OUT) * env.BLOCK_OUT
    stride_h, stride_w = _pair(conv.attrs.strides, "strides")
    pad_top, pad_left, pad_bottom, pad_right = (int(x) for x in conv.attrs.padding)
    if pad_top != pad_bottom or pad_left != pad_right:
        raise ValueError("VTA conv2d currently requires symmetric spatial padding")
    out_height = (height + pad_top + pad_bottom - kernel_h) // stride_h + 1
    out_width = (width + pad_left + pad_right - kernel_w) // stride_w + 1

    data_shape = (
        batch // env.BATCH,
        padded_in_channels // env.BLOCK_IN,
        height,
        width,
        env.BATCH,
        env.BLOCK_IN,
    )
    weight_shape = (
        padded_out_channels // env.BLOCK_OUT,
        padded_in_channels // env.BLOCK_IN,
        kernel_h,
        kernel_w,
        env.BLOCK_OUT,
        env.BLOCK_IN,
    )
    output_shape = (
        batch // env.BATCH,
        padded_out_channels // env.BLOCK_OUT,
        out_height,
        out_width,
        env.BATCH,
        env.BLOCK_OUT,
    )
    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    weight = te.placeholder(weight_shape, name="weight", dtype=env.wgt_dtype)
    bias = te.placeholder(output_shape, name="bias", dtype=env.acc_dtype)
    shortcut = te.placeholder(output_shape, name="shortcut", dtype=env.inp_dtype) if residual else None
    data_buf = topi.nn.pad(
        data, [0, 0, pad_top, pad_left, 0, 0], [0, 0, pad_bottom, pad_right, 0, 0],
        name="data_buf",
    )
    weight_buf = te.compute(weight_shape, lambda *idx: weight(*idx), name="weight_buf")
    ic = te.reduce_axis((0, padded_in_channels // env.BLOCK_IN), name="ic")
    dy = te.reduce_axis((0, kernel_h), name="dy")
    dx = te.reduce_axis((0, kernel_w), name="dx")
    ic_tns = te.reduce_axis((0, env.BLOCK_IN), name="ic_tns")
    accum = te.compute(
        output_shape,
        lambda bo, co, y, x, bi, ci: te.sum(
            data_buf[bo, ic, y * stride_h + dy, x * stride_w + dx, bi, ic_tns].astype(
                env.acc_dtype
            )
            * weight_buf[co, ic, dy, dx, ci, ic_tns].astype(env.acc_dtype),
            axis=[ic, dy, dx, ic_tns],
        ),
        name="accum",
    )
    biased = te.compute(output_shape, lambda *idx: accum(*idx) + bias(*idx), name="biased")
    # Materialize the Q31 multiplier using signed 16-bit ALU immediates, then
    # consume it as the second accumulator operand of a rounded Q31 MUL.
    multiplier_hi = (requant_multiplier + (1 << 15)) >> 16
    if multiplier_hi >= 1 << 15:
        multiplier_hi -= 1 << 16
    multiplier_lo = requant_multiplier - (multiplier_hi << 16)
    multiplier_zero = te.compute(
        output_shape,
        lambda *idx: tvm.tir.call_pure_extern(
            env.acc_dtype, "VTAALUMul", biased(*idx), tvm.tir.const(0, env.acc_dtype)
        ),
        name="requantize_multiplier_zero",
    )
    multiplier_high = te.compute(
        output_shape,
        lambda *idx: multiplier_zero(*idx) + tvm.tir.const(multiplier_hi, env.acc_dtype),
        name="requantize_multiplier_high",
    )
    multiplier_shifted = te.compute(
        output_shape,
        lambda *idx: multiplier_high(*idx) << tvm.tir.const(16, env.acc_dtype),
        name="requantize_multiplier_shifted",
    )
    multiplier_value = te.compute(
        output_shape,
        lambda *idx: multiplier_shifted(*idx) + tvm.tir.const(multiplier_lo, env.acc_dtype),
        name="requantize_multiplier",
    )
    requant_input = biased
    left_shifted = None
    if requant_shift > 0:
        left_shifted = te.compute(
            output_shape,
            lambda *idx: biased(*idx) << tvm.tir.const(requant_shift, env.acc_dtype),
            name="requantize_left_shift",
        )
        requant_input = left_shifted
    high_multiplied = te.compute(
        output_shape,
        lambda *idx: tvm.tir.call_pure_extern(
            env.acc_dtype, "VTAQMultiply", requant_input(*idx), multiplier_value(*idx)
        ),
        name="requantize_high_mul",
    )
    scaled = high_multiplied
    right_shifted = None
    if requant_shift < 0:
        right_shifted = te.compute(
            output_shape,
            lambda *idx: tvm.tir.call_pure_extern(
                env.acc_dtype, "VTARoundingShiftRight", high_multiplied(*idx), -requant_shift
            ),
            name="requantize_right_shift",
        )
        scaled = right_shifted
    shifted = scaled
    if output_zero_point:
        shifted = te.compute(
            output_shape,
            lambda *idx: scaled(*idx) + tvm.tir.const(output_zero_point, env.acc_dtype),
            name="output_zero_point",
        )
    clip_min, clip_max = clip_bounds
    clipped_max = te.compute(
        output_shape,
        lambda *idx: tvm.te.min(shifted(*idx), tvm.tir.const(clip_max, env.acc_dtype)),
        name="clip_max",
    )
    narrowed = te.compute(
        output_shape,
        lambda *idx: tvm.te.max(clipped_max(*idx), tvm.tir.const(clip_min, env.acc_dtype)),
        name="clip_min",
    )
    final_value = narrowed
    if residual:
        shortcut_buf = te.compute(output_shape, lambda *idx: shortcut(*idx), name="shortcut_buf")
        final_value = te.compute(
            output_shape,
            lambda *idx: narrowed(*idx) + shortcut_buf(*idx),
            name="residual_add",
        )
    output = te.compute(output_shape, lambda *idx: final_value(*idx).astype(env.out_dtype), name="output")

    schedule = te.create_schedule(output.op)
    schedule[data_buf].set_scope(env.inp_scope)
    schedule[weight_buf].set_scope(env.wgt_scope)
    requant_stages = ()
    if left_shifted is not None:
        requant_stages += (left_shifted,)
    requant_stages += (multiplier_zero, multiplier_high, multiplier_shifted,
                       multiplier_value)
    requant_stages += (high_multiplied,)
    if right_shifted is not None:
        requant_stages += (right_shifted,)
    accumulator_stages = (accum, biased, clipped_max, narrowed) + requant_stages
    alu_stages = (biased, clipped_max, narrowed) + requant_stages
    if residual:
        accumulator_stages += (shortcut_buf, final_value)
        alu_stages += (final_value,)
    if output_zero_point:
        accumulator_stages += (shifted,)
        alu_stages += (shifted,)
    for stage in accumulator_stages:
        schedule[stage].set_scope(env.acc_scope)
    for stage in alu_stages:
        schedule[stage].pragma(schedule[stage].op.axis[0], env.alu)
    bias_buf = schedule.cache_read(bias, env.acc_scope, [biased])
    schedule[bias_buf].pragma(schedule[bias_buf].op.axis[0], env.dma_copy)
    if residual:
        schedule[shortcut_buf].pragma(schedule[shortcut_buf].op.axis[0], env.dma_copy)
    schedule[data_buf].compute_at(schedule[accum], ic)
    schedule[weight_buf].compute_at(schedule[accum], ic)
    schedule[data_buf].pragma(schedule[data_buf].op.axis[0], env.dma_copy)
    schedule[weight_buf].pragma(schedule[weight_buf].op.axis[0], env.dma_copy)
    bo, co, y, x, bi, ci = schedule[accum].op.axis
    schedule[accum].reorder(ic, bo, co, y, dy, dx, x, bi, ci, ic_tns)
    schedule[accum].tensorize(bi, env.gemm)
    schedule[output].pragma(schedule[output].op.axis[0], env.dma_copy)

    args = [data, weight, bias] + ([shortcut] if residual else []) + [output]
    lowered = lower(schedule, args, simple_mode=True)
    module = build(schedule, args, tvm.target.Target("ext_dev", host=env.target_host), name=name)
    grouped_weight = conv.args[1].data.numpy().astype(env.wgt_dtype)
    raw_weight = np.zeros((out_channels, in_channels, kernel_h, kernel_w), dtype=env.wgt_dtype)
    outputs_per_group = out_channels // groups
    for group in range(groups):
        out_begin = group * outputs_per_group
        in_begin = group * weight_in_per_group
        raw_weight[out_begin : out_begin + outputs_per_group,
                   in_begin : in_begin + weight_in_per_group] = grouped_weight[
                       out_begin : out_begin + outputs_per_group
                   ]
    padded_weight = np.zeros(
        (padded_out_channels, padded_in_channels, kernel_h, kernel_w), dtype=env.wgt_dtype
    )
    padded_weight[:out_channels, :in_channels] = raw_weight
    packed_weight = padded_weight.reshape(
        padded_out_channels // env.BLOCK_OUT,
        env.BLOCK_OUT,
        padded_in_channels // env.BLOCK_IN,
        env.BLOCK_IN,
        kernel_h,
        kernel_w,
    ).transpose(0, 2, 4, 5, 1, 3)
    if raw_bias is None:
        raw_bias = np.zeros((out_channels,), dtype=env.acc_dtype)
    raw_bias = raw_bias.astype(env.acc_dtype) + np.asarray(correction, env.acc_dtype)
    packed_bias = np.broadcast_to(
        np.pad(raw_bias, (0, padded_out_channels - out_channels)).reshape(
            1, padded_out_channels, 1, 1
        ),
        (batch, padded_out_channels, out_height, out_width),
    ).copy()
    packed_bias = packed_bias.reshape(
        batch // env.BATCH,
        env.BATCH,
        padded_out_channels // env.BLOCK_OUT,
        env.BLOCK_OUT,
        out_height,
        out_width,
    ).transpose(0, 2, 4, 5, 1, 3).astype(env.acc_dtype)
    return Conv2DArtifact(
        module,
        lowered,
        packed_weight,
        packed_bias,
        (batch, in_channels, height, width),
        (batch, out_channels, out_height, out_width),
        data_shape,
        output_shape,
    )
