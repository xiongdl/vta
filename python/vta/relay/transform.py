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

"""Function-local Relay legalization and lowering for VTA."""

import tvm
from tvm import relay

from ..environment import get_env
from .contract import COMPILER_NAME, VTACompilerConfig
from .patterns import QNN_CONV2D_COMPOSITE, check_qnn_conv2d


def _is_typed_tensor(expr):
    return isinstance(getattr(expr, "checked_type", None), relay.TensorType)


def _composite_calls(func):
    calls = []

    def visit(node):
        if (
            isinstance(node, relay.Call)
            and isinstance(node.op, relay.Function)
            and node.op.attrs is not None
            and "Composite" in node.op.attrs
        ):
            calls.append(node)

    relay.analysis.post_order_visit(func.body, visit)
    return calls


def _validate_vta_function(func, config=None):
    """Validate an outlined VTA function and return its composite call."""
    if not isinstance(func, relay.Function):
        raise TypeError("func must be a tvm.relay.Function")
    if config is not None and not isinstance(config, VTACompilerConfig):
        raise TypeError("config must be a VTACompilerConfig or None")
    config = config or VTACompilerConfig.from_env(get_env())

    attrs = func.attrs
    if attrs is None or "Compiler" not in attrs or attrs.get_str("Compiler") != COMPILER_NAME:
        raise ValueError("outlined function must have Compiler='vta'")
    if "Primitive" not in attrs or int(attrs["Primitive"]) != 1:
        raise ValueError("outlined VTA function must have Primitive=1")
    if "global_symbol" not in attrs or not attrs.get_str("global_symbol"):
        raise ValueError("outlined VTA function must have a non-empty global_symbol")
    if not all(_is_typed_tensor(param) for param in func.params) or not isinstance(
        func.ret_type, relay.TensorType
    ):
        raise ValueError("outlined VTA function must have inferred tensor types")

    composite_calls = _composite_calls(func)
    if len(composite_calls) != 1:
        raise ValueError("outlined VTA function must contain exactly one composite call")
    composite_call = composite_calls[0]
    composite_name = composite_call.op.attrs.get_str("Composite")
    if composite_name != QNN_CONV2D_COMPOSITE:
        raise ValueError(f"outlined VTA function must contain {QNN_CONV2D_COMPOSITE}")
    if not check_qnn_conv2d(composite_call.op.body, config):
        raise ValueError(
            f"{QNN_CONV2D_COMPOSITE} does not satisfy the active VTA configuration"
        )
    return composite_call


def _static_shape(expr):
    return tuple(int(dim) for dim in expr.checked_type.shape)


def _pack_data(data, shape, config):
    batch, channels, height, width = shape
    reshaped = relay.reshape(
        data,
        (
            batch // config.batch,
            config.batch,
            channels // config.block_in,
            config.block_in,
            height,
            width,
        ),
    )
    return relay.transpose(reshaped, axes=(0, 2, 4, 5, 1, 3))


def _pack_weight(weight, shape, config):
    output_channels, input_channels, height, width = shape
    reshaped = relay.reshape(
        weight,
        (
            output_channels // config.block_out,
            config.block_out,
            input_channels // config.block_in,
            config.block_in,
            height,
            width,
        ),
    )
    return relay.transpose(reshaped, axes=(0, 2, 4, 5, 1, 3))


def _pack_output_constant(constant, config):
    shape = _static_shape(constant)
    if len(shape) == 1:
        constant = relay.reshape(constant, (shape[0], 1, 1))
        shape = (shape[0], 1, 1)
    channels, height, width = shape
    reshaped = relay.reshape(
        constant,
        (channels // config.block_out, config.block_out, height, width, 1),
    )
    transposed = relay.transpose(reshaped, axes=(0, 2, 3, 4, 1))
    return relay.broadcast_to(
        transposed,
        (channels // config.block_out, height, width, config.batch, config.block_out),
    )


def _unpack_data(data, output_shape):
    transposed = relay.transpose(data, axes=(0, 4, 1, 5, 2, 3))
    return relay.reshape(transposed, output_shape)


def _infer_function(func):
    mod = relay.transform.InferType()(tvm.IRModule.from_expr(func))
    return next(iter(mod.functions.values()))


def legalize_vta_function(func, config=None):
    """Rewrite one outlined VTA function into locally packed Relay."""
    config = config or VTACompilerConfig.from_env(get_env())
    composite_call = _validate_vta_function(func, config)
    composite_body = composite_call.op.body
    clipped = composite_body.args[0]
    shifted = clipped.args[0]
    conv_or_bias = shifted.args[0]
    bias = None
    if isinstance(conv_or_bias.op, tvm.ir.Op) and conv_or_bias.op.name in (
        "nn.bias_add",
        "add",
    ):
        conv2d = conv_or_bias.args[0]
        bias = conv_or_bias.args[1]
    else:
        conv2d = conv_or_bias
    if not isinstance(conv2d.op, tvm.ir.Op) or conv2d.op.name != "nn.conv2d":
        raise ValueError("VTA composite does not contain the expected nn.conv2d")

    packed_data = _pack_data(
        composite_call.args[0], _static_shape(composite_call.args[0]), config
    )
    packed_weight = _pack_weight(conv2d.args[1], _static_shape(conv2d.args[1]), config)
    packed_data_layout = f"NCHW{config.batch}n{config.block_in}c"
    packed_output_layout = f"NCHW{config.batch}n{config.block_out}c"
    packed_kernel_layout = f"OIHW{config.block_out}o{config.block_in}i"
    packed_conv2d = relay.nn.conv2d(
        packed_data,
        packed_weight,
        strides=conv2d.attrs.strides,
        padding=conv2d.attrs.padding,
        dilation=conv2d.attrs.dilation,
        groups=conv2d.attrs.groups,
        channels=conv2d.attrs.channels,
        kernel_size=conv2d.attrs.kernel_size,
        data_layout=packed_data_layout,
        kernel_layout=packed_kernel_layout,
        out_layout=packed_output_layout,
        out_dtype=conv2d.attrs.out_dtype,
    )
    packed_accumulator = packed_conv2d
    if bias is not None:
        packed_accumulator = relay.add(
            packed_accumulator, _pack_output_constant(bias, config)
        )
    packed_shifted = relay.right_shift(packed_accumulator, shifted.args[1])
    packed_clipped = relay.clip(packed_shifted, clipped.attrs.a_min, clipped.attrs.a_max)
    packed_cast = relay.cast(packed_clipped, composite_body.attrs.dtype)
    unpacked = _unpack_data(packed_cast, _static_shape(composite_body))
    legalized = relay.Function(
        func.params,
        unpacked,
        func.ret_type,
        func.type_params,
        func.attrs,
    )
    return _infer_function(legalized)
