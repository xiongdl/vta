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
from tvm.relay.op.contrib import get_pattern_table

from byoc_utils import make_qnn_conv2d_module, make_qnn_conv2d_near_miss_module
from vta.relay.contract import VTACompilerConfig
from vta.relay.patterns import QNN_CONV2D_COMPOSITE, check_qnn_conv2d, pattern_table


def _merge_composites(mod):
    return relay.transform.MergeComposite(pattern_table())(mod)


def _composite_functions(expr):
    functions = []

    def visit(node):
        if isinstance(node, relay.Function) and node.attrs is not None:
            if node.attrs.get_str("Composite") == QNN_CONV2D_COMPOSITE:
                functions.append(node)

    relay.analysis.post_order_visit(expr, visit)
    return functions


def _root_call(mod):
    return mod["main"].body.args[0]


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_qnn_conv2d_pattern_matches_approved_forms(bias_kind):
    mod = make_qnn_conv2d_module(vta.get_env(), bias_kind=bias_kind)

    merged = _merge_composites(mod)

    assert len(_composite_functions(merged["main"].body)) == 1


def test_qnn_conv2d_pattern_keeps_host_operations_outside_composite():
    mod = make_qnn_conv2d_module(vta.get_env())

    merged = _merge_composites(mod)
    composite = _composite_functions(merged["main"].body)[0]

    assert isinstance(merged["main"].body.op, tvm.ir.Op)
    assert merged["main"].body.op.name == "transpose"
    assert isinstance(composite.body, relay.Call)
    assert composite.body.op.name == "cast"
    assert "abs" not in composite.astext(show_meta_data=False)
    assert "transpose" not in composite.astext(show_meta_data=False)


def test_pattern_module_import_does_not_register_global_table():
    assert get_pattern_table("vta") is None


def test_qnn_conv2d_predicate_rejects_non_constant_weight():
    mod, _ = make_qnn_conv2d_near_miss_module(vta.get_env())

    assert not check_qnn_conv2d(_root_call(mod))


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_dtype": "int16"},
        {"weight_dtype": "int16"},
        {"accumulator_dtype": "int16"},
        {"output_dtype": "int16"},
    ],
)
def test_qnn_conv2d_predicate_rejects_wrong_dtype(overrides):
    mod = make_qnn_conv2d_module(vta.get_env(), **overrides)

    assert not check_qnn_conv2d(
        _root_call(mod), VTACompilerConfig.from_env(vta.get_env())
    )


def test_qnn_conv2d_predicate_accepts_approved_hardware_domain():
    mod = make_qnn_conv2d_module(vta.get_env())

    assert check_qnn_conv2d(_root_call(mod))


@pytest.mark.parametrize(
    "overrides",
    [
        {"kernel_size": (1, 1), "padding": (0, 0)},
        {"strides": (2, 2)},
        {"dilation": (2, 2)},
        {"padding": (0, 0)},
        {"shift": -1},
        {"shift": 32},
        {"clip_bounds": (-129, 127)},
        {"clip_bounds": (-128, 128)},
        {"input_channels": 8},
        {"output_channels": 8},
        {"data_layout": "NHWC", "kernel_layout": "HWIO"},
        {"out_layout": "NHWC"},
        {"groups": 2},
    ],
)
def test_qnn_conv2d_predicate_rejects_unsupported_hardware_boundary(overrides):
    mod = make_qnn_conv2d_module(vta.get_env(), **overrides)

    assert not check_qnn_conv2d(_root_call(mod))
