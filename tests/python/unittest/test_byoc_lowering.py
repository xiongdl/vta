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

from dataclasses import replace

import pytest
import tvm
import vta
from tvm import relay

from byoc_utils import make_qnn_conv2d_module
from vta.relay import partition_for_vta
from vta.relay.contract import VTACompilerConfig
from vta.relay.transform import _validate_vta_function


def _partitioned_function():
    mod = partition_for_vta(make_qnn_conv2d_module(vta.get_env()), mod_name="lowering")
    return next(
        function
        for function in mod.functions.values()
        if isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
        and function.attrs.get_str("Compiler") == "vta"
    )


def test_validate_vta_function_accepts_approved_partition():
    external = _partitioned_function()

    composite_call = _validate_vta_function(external)

    assert isinstance(composite_call, relay.Call)
    assert composite_call.op.attrs.get_str("Composite") == "vta.qnn_conv2d"


@pytest.mark.parametrize("invalid_func", [None, tvm.IRModule(), relay.var("data")])
def test_validate_vta_function_rejects_wrong_python_type(invalid_func):
    with pytest.raises(TypeError, match="func must be a tvm.relay.Function"):
        _validate_vta_function(invalid_func)


def test_validate_vta_function_rejects_wrong_config_type():
    with pytest.raises(TypeError, match="config must be a VTACompilerConfig or None"):
        _validate_vta_function(_partitioned_function(), config={})


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("Compiler", "cpu", "Compiler='vta'"),
        ("Primitive", 0, "Primitive=1"),
        ("global_symbol", "", "non-empty global_symbol"),
    ],
)
def test_validate_vta_function_rejects_invalid_external_attribute(attribute, value, message):
    external = _partitioned_function().with_attr(attribute, value)

    with pytest.raises(ValueError, match=message):
        _validate_vta_function(external)


def test_validate_vta_function_rejects_missing_external_attributes():
    external = _partitioned_function()
    unannotated = relay.Function(external.params, external.body, external.ret_type)

    with pytest.raises(ValueError, match="Compiler='vta'"):
        _validate_vta_function(unannotated)


def test_validate_vta_function_rejects_unsupported_composite_name():
    external = _partitioned_function()
    composite_call = external.body
    unsupported = composite_call.op.with_attr("Composite", "vta.unsupported")
    malformed = relay.Function(
        external.params,
        relay.Call(unsupported, composite_call.args),
        external.ret_type,
        attrs=external.attrs,
    )

    with pytest.raises(ValueError, match="vta.qnn_conv2d"):
        _validate_vta_function(malformed)


def test_validate_vta_function_rejects_malformed_composite_body():
    external = _partitioned_function()
    composite_call = external.body
    composite = composite_call.op
    malformed_composite = relay.Function(
        composite.params,
        relay.abs(composite.params[0]),
        attrs=composite.attrs,
    )
    malformed = relay.Function(
        external.params,
        relay.Call(malformed_composite, composite_call.args),
        external.ret_type,
        attrs=external.attrs,
    )
    malformed_mod = relay.transform.InferType()(tvm.IRModule.from_expr(malformed))
    malformed = next(iter(malformed_mod.functions.values()))

    with pytest.raises(ValueError, match="does not satisfy the active VTA configuration"):
        _validate_vta_function(malformed)


def test_validate_vta_function_rejects_config_mismatch():
    external = _partitioned_function()
    config = VTACompilerConfig.from_env(vta.get_env())
    incompatible = replace(config, block_in=config.block_in * 2)

    with pytest.raises(ValueError, match="does not satisfy the active VTA configuration"):
        _validate_vta_function(external, incompatible)


def test_importing_lowering_module_does_not_register_external_compiler():
    assert tvm.get_global_func("relay.ext.vta", allow_missing=True) is None
