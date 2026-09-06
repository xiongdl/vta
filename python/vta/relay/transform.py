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
