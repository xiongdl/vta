# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Relay dataflow patterns supported by the VTA external compiler."""

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import is_constant, is_op, wildcard

from ..environment import get_env
from .contract import VTACompilerConfig


QNN_CONV2D_COMPOSITE = "vta.qnn_conv2d"


def qnn_conv2d_pattern():
    """Match the initial VTA quantized convolution composite."""
    conv2d = is_op("nn.conv2d")(wildcard(), is_constant())
    with_bias = conv2d | is_op("nn.bias_add")(conv2d, is_constant())
    with_bias = with_bias | is_op("add")(conv2d, is_constant())
    shifted = is_op("right_shift")(with_bias, is_constant())
    clipped = is_op("clip")(shifted)
    return is_op("cast")(clipped)


def _is_call(call, operator_name):
    return (
        isinstance(call, relay.Call)
        and isinstance(call.op, tvm.ir.Op)
        and call.op.name == operator_name
    )


def _tensor_dtype(expr):
    checked_type = getattr(expr, "checked_type", None)
    if not isinstance(checked_type, relay.TensorType):
        return None
    return checked_type.dtype


def _static_shape(expr):
    checked_type = getattr(expr, "checked_type", None)
    if not isinstance(checked_type, relay.TensorType):
        return None
    if not all(isinstance(dim, tvm.tir.IntImm) for dim in checked_type.shape):
        return None
    return tuple(int(dim) for dim in checked_type.shape)


def _scalar_integer(constant):
    if not isinstance(constant, relay.Constant):
        return None
    values = constant.data.numpy()
    if values.size != 1 or values.dtype.kind not in "iu":
        return None
    return int(values.item())


def _hardware_domain_is_supported(conv2d, shifted, clipped, config):
    data, weight = conv2d.args
    data_shape = _static_shape(data)
    weight_shape = _static_shape(weight)
    output_shape = _static_shape(conv2d)
    if data_shape is None or weight_shape is None or output_shape is None:
        return False
    if len(data_shape) != 4 or len(weight_shape) != 4 or len(output_shape) != 4:
        return False

    attrs = conv2d.attrs
    data_layout = str(attrs.data_layout)
    kernel_layout = str(attrs.kernel_layout)
    if (data_layout, kernel_layout) not in (("NCHW", "OIHW"), ("NHWC", "HWIO")):
        return False
    output_layout = str(attrs.out_layout) or data_layout
    if output_layout != data_layout:
        return False
    kernel_size = tuple(int(value) for value in attrs.kernel_size)
    if kernel_size not in ((1, 1), (3, 3)):
        return False
    strides = tuple(int(value) for value in attrs.strides)
    if strides not in ((1, 1), (2, 2)):
        return False
    if tuple(int(value) for value in attrs.dilation) != (1, 1) or int(attrs.groups) != 1:
        return False
    padding = tuple(int(value) for value in attrs.padding)
    if len(padding) != 4 or any(value < 0 for value in padding):
        return False

    if data_layout == "NCHW":
        batch, input_channels, input_height, input_width = data_shape
        output_batch, output_channels, output_height, output_width = output_shape
        expected_weight_shape = (output_channels, input_channels, *kernel_size)
    else:
        batch, input_height, input_width, input_channels = data_shape
        output_batch, output_height, output_width, output_channels = output_shape
        expected_weight_shape = (*kernel_size, input_channels, output_channels)

    if batch <= 0 or batch != output_batch or batch % config.batch != 0:
        return False
    if input_channels <= 0 or input_channels % config.block_in != 0:
        return False
    if output_channels <= 0 or output_channels % config.block_out != 0:
        return False
    if int(attrs.channels) != output_channels:
        return False
    if weight_shape != expected_weight_shape:
        return False
    padded_height = input_height + padding[0] + padding[2]
    padded_width = input_width + padding[1] + padding[3]
    expected_height = (padded_height - kernel_size[0]) // strides[0] + 1
    expected_width = (padded_width - kernel_size[1]) // strides[1] + 1
    if expected_height <= 0 or expected_width <= 0:
        return False
    if (output_height, output_width) != (expected_height, expected_width):
        return False

    shift = _scalar_integer(shifted.args[1])
    accumulator_bits = tvm.DataType(config.accumulator_dtype).bits
    if shift is None or shift < 0 or shift >= accumulator_bits:
        return False

    output_type = tvm.DataType(config.output_dtype)
    if output_type.type_code == tvm.DataType("int8").type_code:
        minimum = -(1 << (output_type.bits - 1))
        maximum = (1 << (output_type.bits - 1)) - 1
    else:
        minimum = 0
        maximum = (1 << output_type.bits) - 1
    return float(clipped.attrs.a_min) >= minimum and float(clipped.attrs.a_max) <= maximum


def check_qnn_conv2d(call, config=None):
    """Return whether a matched convolution has the required constants and dtypes."""
    config = config or VTACompilerConfig.from_env(get_env())
    if not _is_call(call, "cast") or _tensor_dtype(call) != config.output_dtype:
        return False

    clipped = call.args[0]
    if not _is_call(clipped, "clip"):
        return False
    shifted = clipped.args[0]
    if not _is_call(shifted, "right_shift") or not isinstance(
        shifted.args[1], relay.Constant
    ):
        return False

    conv_or_bias = shifted.args[0]
    bias = None
    if _is_call(conv_or_bias, "nn.bias_add") or _is_call(conv_or_bias, "add"):
        bias = conv_or_bias.args[1]
        conv2d = conv_or_bias.args[0]
    else:
        conv2d = conv_or_bias

    if not _is_call(conv2d, "nn.conv2d") or len(conv2d.args) != 2:
        return False
    data, weight = conv2d.args
    if not isinstance(weight, relay.Constant):
        return False
    if bias is not None and not isinstance(bias, relay.Constant):
        return False

    return (
        _tensor_dtype(data) == config.input_dtype
        and _tensor_dtype(weight) == config.weight_dtype
        and _tensor_dtype(conv2d) == config.accumulator_dtype
        and str(conv2d.attrs.out_dtype) == config.accumulator_dtype
        and (bias is None or _tensor_dtype(bias) == config.accumulator_dtype)
        and _hardware_domain_is_supported(conv2d, shifted, clipped, config)
    )


def pattern_table(config=None):
    """Return VTA composite patterns without registering global state."""
    config = config or VTACompilerConfig.from_env(get_env())
    return [
        (
            QNN_CONV2D_COMPOSITE,
            qnn_conv2d_pattern(),
            lambda call: check_qnn_conv2d(call, config),
        )
    ]
