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

"""Shared Relay fixtures for VTA BYOC tests."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

import tvm
from tvm import relay


def run_isolated_python(source, *, env=None):
    """Run source in a clean Python process with the test helpers importable."""
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    test_dir = str(Path(__file__).resolve().parent)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {test_dir!r})\n{textwrap.dedent(source)}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=process_env,
    )


def _make_qnn_conv2d_module(
    env,
    constant_weights,
    bias_kind=None,
    input_dtype=None,
    weight_dtype=None,
    accumulator_dtype=None,
    output_dtype=None,
    bias_dtype=None,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding=(1, 1),
    dilation=(1, 1),
    shift=1,
    clip_bounds=(-128, 127),
    input_channels=None,
    output_channels=None,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    groups=1,
    batch=None,
    input_height=8,
    input_width=8,
):
    input_dtype = input_dtype or env.inp_dtype
    weight_dtype = weight_dtype or env.wgt_dtype
    accumulator_dtype = accumulator_dtype or env.acc_dtype
    output_dtype = output_dtype or env.out_dtype
    bias_dtype = bias_dtype or accumulator_dtype
    input_channels = env.BLOCK_IN if input_channels is None else input_channels
    output_channels = env.BLOCK_OUT if output_channels is None else output_channels
    batch = env.BATCH if batch is None else batch
    input_shape = (batch, input_channels, input_height, input_width)
    if data_layout == "NHWC":
        input_shape = (batch, input_height, input_width, input_channels)
    weight_shape = (output_channels, input_channels // groups, *kernel_size)
    if kernel_layout == "HWIO":
        weight_shape = (*kernel_size, input_channels, output_channels)
    data = relay.var("data", shape=input_shape, dtype=input_dtype)
    host_pre = relay.abs(data)

    params = [data]
    if constant_weights:
        weight = relay.const(np.ones(weight_shape, dtype=weight_dtype))
    else:
        weight = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
        params.append(weight)

    conv = relay.nn.conv2d(
        host_pre,
        weight,
        channels=output_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        out_layout=out_layout,
        groups=groups,
        out_dtype=accumulator_dtype,
    )
    if bias_kind == "bias_add":
        bias = relay.const(np.ones((output_channels,), dtype=bias_dtype))
        axis = 3 if data_layout == "NHWC" else 1
        conv = relay.nn.bias_add(conv, bias, axis=axis)
    elif bias_kind == "add":
        bias_shape = (1, 1, output_channels) if data_layout == "NHWC" else (
            output_channels,
            1,
            1,
        )
        bias = relay.const(np.ones(bias_shape, dtype=bias_dtype))
        conv = relay.add(conv, bias)
    elif bias_kind is not None:
        raise ValueError(f"unsupported bias kind: {bias_kind}")

    shifted = relay.right_shift(conv, relay.const(shift, accumulator_dtype))
    clipped = relay.clip(shifted, a_min=clip_bounds[0], a_max=clip_bounds[1])
    narrowed = relay.cast(clipped, output_dtype)
    host_post_axes = (0, 3, 1, 2) if data_layout == "NHWC" else (0, 2, 3, 1)
    host_post = relay.transpose(narrowed, axes=host_post_axes)

    mod = tvm.IRModule.from_expr(relay.Function(params, host_post))
    return relay.transform.InferType()(mod)


def make_qnn_conv2d_module(env, bias_kind=None, **dtype_overrides):
    """Build the initial supported VTA BYOC convolution fixture."""
    return _make_qnn_conv2d_module(
        env, constant_weights=True, bias_kind=bias_kind, **dtype_overrides
    )


def make_qnn_conv2d_near_miss_module(env, **overrides):
    """Build a fixture rejected because its convolution weight is not constant."""
    return (
        _make_qnn_conv2d_module(env, constant_weights=False, **overrides),
        "constant_weights",
    )


def make_adjacent_qnn_conv2d_module(env, *, count=2, **overrides):
    """Build directly adjacent supported convolution tails for partition tests."""
    if count < 2:
        raise ValueError("count must be at least two")

    data_layout = overrides.pop("data_layout", "NCHW")
    kernel_layout = overrides.pop("kernel_layout", "OIHW")
    kernel_size = overrides.pop("kernel_size", (3, 3))
    strides = overrides.pop("strides", (1, 1))
    padding = overrides.pop("padding", (1, 1))
    if overrides:
        raise ValueError(f"unsupported adjacent fixture options: {sorted(overrides)}")

    input_shape = (env.BATCH, env.BLOCK_IN, 8, 8)
    if data_layout == "NHWC":
        input_shape = (env.BATCH, 8, 8, env.BLOCK_IN)
    data = relay.var("data", shape=input_shape, dtype=env.inp_dtype)
    value = relay.abs(data)
    for _ in range(count):
        weight_shape = (env.BLOCK_OUT, env.BLOCK_IN, *kernel_size)
        if kernel_layout == "HWIO":
            weight_shape = (*kernel_size, env.BLOCK_IN, env.BLOCK_OUT)
        weight = relay.const(np.ones(weight_shape, dtype=env.wgt_dtype))
        value = relay.nn.conv2d(
            value,
            weight,
            channels=env.BLOCK_OUT,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            out_dtype=env.acc_dtype,
        )
        value = relay.right_shift(value, relay.const(1, env.acc_dtype))
        value = relay.clip(value, a_min=-128, a_max=127)
        value = relay.cast(value, env.out_dtype)

    host_post_axes = (0, 3, 1, 2) if data_layout == "NHWC" else (0, 2, 3, 1)
    value = relay.transpose(value, axes=host_post_axes)
    mod = tvm.IRModule.from_expr(relay.Function([data], value))
    return relay.transform.InferType()(mod)
