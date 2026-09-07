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

from .contract import COMPILER_NAME
from .transform import lower_vta_function


def _compile_vta_function(func):
    """Compile one outlined VTA Relay function into a runtime module."""
    if not isinstance(func, relay.Function):
        raise TypeError("func must be a tvm.relay.Function")

    # TECompiler removes Compiler immediately before invoking relay.ext.vta.
    # Restore the hook identity locally so the lowering boundary can perform
    # its complete validation without weakening direct callers.
    if func.attrs is None or "Compiler" not in func.attrs:
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
