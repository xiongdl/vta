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

"""Migration contracts for retiring the legacy VTA graph-pack entry point."""

import ast
import importlib.util
from pathlib import Path

import pytest
import tvm
import vta
from tvm import relay

from byoc_utils import make_qnn_conv2d_module


VTA_ROOT = Path(__file__).resolve().parents[3]
CONSUMERS = (
    VTA_ROOT / "apps" / "deploy" / "resnet_export.py",
    VTA_ROOT / "tutorials" / "frontend" / "deploy_detection.py",
)
LEGACY_NAMES = {
    "graph_pack",
    "get_subgraph",
    "start_name",
    "stop_name",
    "start_name_idx",
    "stop_name_idx",
    "bitpack_start",
    "bitpack_end",
}


@pytest.mark.parametrize("consumer", CONSUMERS, ids=lambda path: path.stem)
def test_consumer_uses_capability_partitioning_without_graphpack_ranges(consumer):
    source = consumer.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(consumer))
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    keywords = {node.arg for node in ast.walk(tree) if isinstance(node, ast.keyword)}

    assert "register_byoc" in attributes
    assert "partition_for_vta" in attributes
    assert not LEGACY_NAMES.intersection(identifiers | attributes | keywords)


@pytest.mark.parametrize("mod_name", ("resnet18_v1", "yolov3_tiny"))
def test_consumer_fixture_creates_and_builds_a_vta_partition(mod_name):
    env = vta.get_env()
    partitioned = vta.relay.partition_for_vta(
        make_qnn_conv2d_module(env), mod_name=mod_name
    )

    assert any(
        isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
        and function.attrs.get_str("Compiler") == "vta"
        for function in partitioned.functions.values()
    )

    vta.register_byoc()
    with vta.build_config():
        factory = relay.build(
            partitioned,
            target=tvm.target.Target(env.target, host=env.target_host),
        )
    assert factory.get_lib() is not None


def test_graphpack_public_api_and_implementation_are_removed():
    assert not hasattr(vta.top, "graph_pack")
    assert importlib.util.find_spec("vta.top.graphpack") is None
    assert not (VTA_ROOT / "python" / "vta" / "top" / "graphpack.py").exists()
