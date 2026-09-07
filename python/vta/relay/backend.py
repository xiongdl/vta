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

"""Runtime artifact generation for the VTA Relay external compiler."""

import tvm
from tvm import relay

from .contract import COMPILER_NAME, EXTERNAL_COMPILER
from .transform import lower_vta_function


_REGISTERED_COMPILER = None


def _compile_vta_function(func):
    """Compile one outlined VTA Relay function into a runtime module."""
    if not isinstance(func, relay.Function):
        raise TypeError("func must be a tvm.relay.Function")

    # TECompiler removes Compiler immediately before invoking relay.ext.vta.
    # Restore the hook identity locally so the lowering boundary can perform
    # its complete validation without weakening direct callers.
    compiler = (
        func.attrs.get_str("Compiler")
        if func.attrs is not None and "Compiler" in func.attrs
        else None
    )
    if compiler is None:
        func = func.with_attr("Compiler", COMPILER_NAME)

    primfunc = lower_vta_function(func)
    symbol = func.attrs.get_str("global_symbol")
    module = tvm.build(
        tvm.IRModule({symbol: primfunc}),
        target=primfunc.attrs["target"],
    )
    if not isinstance(module, tvm.runtime.Module) or module.handle.value is None:
        raise RuntimeError(f"VTA compilation returned no runtime module for {symbol}")
    if not module.implements_function(symbol, True):
        raise RuntimeError(f"VTA runtime module does not implement {symbol}")
    return module


def register_byoc():
    """Register the VTA Relay external compiler in the current TVM process."""
    global _REGISTERED_COMPILER

    existing = tvm.get_global_func(EXTERNAL_COMPILER, allow_missing=True)
    if _REGISTERED_COMPILER is not None:
        if existing is not None and existing.handle.value == _REGISTERED_COMPILER.handle.value:
            return
        raise RuntimeError(f"{EXTERNAL_COMPILER} is already registered by another callback")
    if existing is not None:
        raise RuntimeError(f"{EXTERNAL_COMPILER} is already registered by another callback")

    try:
        _REGISTERED_COMPILER = tvm.register_func(
            EXTERNAL_COMPILER,
            _compile_vta_function,
        )
    except tvm.error.TVMError as err:
        raise RuntimeError(
            f"{EXTERNAL_COMPILER} is already registered by another callback"
        ) from err
