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

import json

import numpy as np
import pytest
import tvm
import vta
from tvm import relay, rpc
from tvm.contrib import graph_executor, utils

from byoc_utils import make_qnn_conv2d_module
from vta.relay import partition_for_vta
from vta.testing import simulator


def _run_graph(factory, device, input_data):
    runtime = graph_executor.GraphModule(factory["default"](device))
    runtime.set_input("data", input_data)
    runtime.run()
    return runtime.get_output(0).numpy()


def _remote_simulator_stats(remote):
    status = remote.get_function("vta.simulator.profiler_status")
    return json.loads(status())


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_exported_approved_graph_executes_on_fsim(bias_kind):
    env = vta.get_env()
    if not simulator.enabled():
        raise RuntimeError(
            "VTA FSIM is unavailable; run "
            "./scripts/build_vta_lib.sh --target libvta_fsim"
        )
    mod = make_qnn_conv2d_module(env, bias_kind=bias_kind)
    input_shape = tuple(int(dim) for dim in mod["main"].params[0].checked_type.shape)
    input_data = ((np.arange(np.prod(input_shape)) % 17) - 8).reshape(input_shape)
    input_data = input_data.astype(env.inp_dtype)

    reference_factory = relay.build(mod, target="llvm")
    expected = _run_graph(reference_factory, tvm.cpu(0), input_data)

    partitioned = partition_for_vta(mod, mod_name="runtime")
    external_functions = [
        function
        for function in partitioned.functions.values()
        if isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
    ]
    assert len(external_functions) == 1
    external = external_functions[0]
    symbol = external.attrs.get_str("global_symbol")
    assert len(external.params) == 1
    assert "abs" in partitioned["main"].astext(show_meta_data=False)
    assert "transpose" in partitioned["main"].astext(show_meta_data=False)

    simulator.clear_stats()
    vta.register_byoc()
    with vta.build_config():
        factory = relay.build(
            partitioned,
            target=tvm.target.Target(env.target, host=env.target_host),
        )
    assert all(counter == 0 for counter in simulator.stats().values())

    graph = json.loads(factory.get_graph_json())
    graph_inputs = [graph["nodes"][index]["name"] for index in graph["arg_nodes"]]
    assert graph_inputs == ["data"]

    artifact_dir = utils.tempdir()
    artifact_name = f"vta_byoc_runtime_{bias_kind or 'no_bias'}.tar"
    artifact_path = artifact_dir.relpath(artifact_name)
    factory.export_library(artifact_path)

    remote = rpc.LocalSession()
    remote.upload(artifact_path)
    loaded = remote.load_module(artifact_name)
    assert loaded.implements_function(symbol, True)

    remote.get_function("vta.simulator.profiler_clear")()
    runtime = graph_executor.create(factory.get_graph_json(), loaded, remote.ext_dev(0))
    runtime.set_input("data", input_data)
    assert all(counter == 0 for counter in _remote_simulator_stats(remote).values())
    runtime.run()
    actual = runtime.get_output(0).numpy()
    runtime_stats = _remote_simulator_stats(remote)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    assert runtime_stats["gemm_counter"] > 0
    assert runtime_stats["wgt_load_nbytes"] > 0
    assert runtime_stats["out_store_nbytes"] > 0
