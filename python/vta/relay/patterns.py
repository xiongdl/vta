"""Dataflow patterns and capability predicates for VTA."""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import is_constant, is_op, wildcard

from ..config import VTAConfig


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
    input_scale, kernel_scale = (_scalar(core.args[i]) for i in range(4, 6))
    requant_input_scale = _scalar(requantize.args[1])
    requant_output_scale = _scalar(requantize.args[3])
    values = (input_scale, kernel_scale, requant_input_scale, requant_output_scale)
    if input_zero_point != 0 or kernel_zero_point != 0 or any(x is None for x in values):
        return False
    if not np.isclose(input_scale * kernel_scale, requant_input_scale):
        return False
    if not np.isclose(requant_input_scale, requant_output_scale):
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
    if _as_int(core.attrs.groups) != 1:
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


def pattern_table(config=None):
    """Return VTA composite patterns without importing TVM's legacy graphpack."""
    config = config or VTAConfig.from_json()
    return [
        ("vta.qnn_conv2d", _qnn_conv2d_pattern(), lambda call: _check_common_qnn(call, config)),
        (
            "vta.qnn_dense",
            _qnn_dense_pattern(),
            lambda call: _check_common_qnn(call, config, dense=True),
        ),
    ]
