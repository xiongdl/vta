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
