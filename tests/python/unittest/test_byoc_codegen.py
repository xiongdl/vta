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

import pytest
import tvm
import vta
from tvm import relay

from byoc_utils import make_qnn_conv2d_module
from vta.relay import partition_for_vta
from vta.relay.transform import lower_vta_function


def _partitioned_function(bias_kind=None):
    mod = partition_for_vta(
        make_qnn_conv2d_module(vta.get_env(), bias_kind=bias_kind),
        mod_name="codegen",
    )
    return next(
        function
        for function in mod.functions.values()
        if isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
    )


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_lowered_vta_function_builds_with_internal_constants(bias_kind):
    external = _partitioned_function(bias_kind)
    primfunc = lower_vta_function(external)
    symbol = external.attrs.get_str("global_symbol")

    module = tvm.build(
        tvm.IRModule({symbol: primfunc}),
        target=primfunc.attrs["target"],
    )

    assert isinstance(module, tvm.runtime.Module)
    assert module.handle.value is not None
    assert len(primfunc.params) == 2
