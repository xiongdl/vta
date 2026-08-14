"""Dataflow patterns and capability predicates for VTA."""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import is_constant, is_op, wildcard

from ..config import VTAConfig
from .quantization import quantize_multiplier


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dtype_bits(dtype):
    return tvm.DataType(dtype).bits


def _find_call(expr, op_name):
    if isinstance(expr, relay.Call):
        if isinstance(expr.op, tvm.ir.Op) and expr.op.name == op_name:
            return expr
        for arg in expr.args:
            found = _find_call(arg, op_name)
            if found is not None:
                return found
    return None


def _scalar(expr):
    if not isinstance(expr, relay.Constant) or expr.data.numpy().size != 1:
        return None
    return expr.data.numpy().item()


def _array(expr):
    if not isinstance(expr, relay.Constant):
        return None
    return np.asarray(expr.data.numpy())


def _check_common_qnn(call, config, dense=False):
    op_name = "qnn.dense" if dense else "qnn.conv2d"
    core = _find_call(call, op_name)
    if core is None or len(core.args) < 2 or not isinstance(core.args[1], relay.Constant):
        return False
    data_type = core.args[0].checked_type
    weight_type = core.args[1].checked_type
    if _dtype_bits(data_type.dtype) != config.input_bits:
        return False
    if _dtype_bits(weight_type.dtype) != config.weight_bits:
        return False
    requantize = _find_call(call, "qnn.requantize")
    if requantize is None or requantize.checked_type.dtype != "int8":
        return False
    input_zero_point, kernel_zero_point = (_scalar(core.args[i]) for i in range(2, 4))
    input_scale, kernel_scale = (_array(core.args[i]) for i in range(4, 6))
    requant_input_scale = _array(requantize.args[1])
    requant_output_scale = _array(requantize.args[3])
    values = (input_scale, kernel_scale, requant_input_scale, requant_output_scale)
    if input_zero_point != 0 or kernel_zero_point != 0 or any(x is None for x in values):
        return False
    if not np.allclose(input_scale * kernel_scale, requant_input_scale):
        return False
    try:
        shifts = np.broadcast_to(requant_input_scale, np.broadcast_shapes(
            requant_input_scale.shape, requant_output_scale.shape
        )) / np.broadcast_to(requant_output_scale, np.broadcast_shapes(
            requant_input_scale.shape, requant_output_scale.shape
        ))
        if dense and not np.allclose(shifts, 1.0):
            return False
        [quantize_multiplier(value) for value in shifts.reshape(-1)]
    except (ValueError, TypeError):
        return False
    if any(
        not isinstance(dim, tvm.tir.IntImm)
        for dim in list(data_type.shape) + list(weight_type.shape)
    ):
        return False
    if dense:
        if len(data_type.shape) != 2 or len(weight_type.shape) != 2:
            return False
        return int(data_type.shape[0]) % config.batch == 0
    if core.attrs.data_layout != "NCHW" or core.attrs.kernel_layout != "OIHW":
        return False
    if tuple(int(x) for x in core.attrs.dilation) != (1, 1):
        return False
    groups = _as_int(core.attrs.groups)
    if groups is None or groups <= 0:
        return False
    in_channels = int(data_type.shape[1])
    out_channels, weight_in = int(weight_type.shape[0]), int(weight_type.shape[1])
    if in_channels % groups or out_channels % groups or weight_in * groups != in_channels:
        return False
    padding = tuple(int(x) for x in core.attrs.padding)
    if len(padding) != 4 or padding[0] != padding[2] or padding[1] != padding[3]:
        return False
    if any(int(x) <= 0 for x in core.attrs.strides):
        return False
    return len(data_type.shape) == 4 and len(weight_type.shape) == 4


def _qnn_conv2d_pattern():
    conv = is_op("qnn.conv2d")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    biased = is_op("nn.bias_add")(conv, is_constant())
    requantize = is_op("qnn.requantize")(
        conv | biased, is_constant(), is_constant(), is_constant(), is_constant()
    )
    return requantize.optional(is_op("clip"))


def _qnn_dense_pattern():
    dense = is_op("qnn.dense")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    biased = is_op("nn.bias_add")(dense, is_constant())
    requantize = is_op("qnn.requantize")(
        dense | biased, is_constant(), is_constant(), is_constant(), is_constant()
    )
    return requantize.optional(is_op("clip"))


def _check_add(call, config):
    if len(call.args) != 2:
        return False
    # Keep constant post-processing adds on LLVM.  The VTA ALU region is
    # intended for a genuine residual edge with two runtime tensors.
    if any(isinstance(arg, relay.Constant) for arg in call.args):
        return False
    lhs, rhs = call.args[0].checked_type, call.args[1].checked_type
    return lhs.dtype == "int8" and rhs.dtype == "int8" and list(lhs.shape) == list(rhs.shape) and all(
        isinstance(dim, tvm.tir.IntImm) for dim in lhs.shape
    )


def pattern_table(config=None):
    """Return VTA composite patterns without importing TVM's legacy graphpack."""
    config = config or VTAConfig.from_json()
    return [
        ("vta.add", is_op("add")(wildcard(), wildcard()), lambda call: _check_add(call, config)),
        ("vta.qnn_conv2d", _qnn_conv2d_pattern(), lambda call: _check_common_qnn(call, config)),
        (
            "vta.qnn_dense",
            _qnn_dense_pattern(),
            lambda call: _check_common_qnn(call, config, dense=True),
        ),
    ]
