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

from byoc_utils import make_qnn_conv2d_module, make_qnn_conv2d_near_miss_module
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


def _require_fsim():
    if not simulator.enabled():
        raise RuntimeError(
            "VTA FSIM is unavailable; run "
            "./scripts/build_vta_lib.sh --target libvta_fsim"
        )


def _require_runtime_symbol(module, symbol):
    if not module.implements_function(symbol, True):
        raise RuntimeError(f"loaded VTA artifact does not implement {symbol}")


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_exported_approved_graph_executes_on_fsim(bias_kind):
    env = vta.get_env()
    _require_fsim()
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
    _require_runtime_symbol(loaded, symbol)

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


def test_near_miss_executes_only_on_host():
    env = vta.get_env()
    mod, _ = make_qnn_conv2d_near_miss_module(env)
    partitioned = partition_for_vta(mod)

    assert not any(
        isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
        for function in partitioned.functions.values()
    )

    factory = relay.build(partitioned, target="llvm")
    graph = json.loads(factory.get_graph_json())
    assert set(graph["attrs"]["device_index"][1]) == {tvm.cpu(0).device_type}

    data_shape = tuple(int(dim) for dim in mod["main"].params[0].checked_type.shape)
    weight_shape = tuple(int(dim) for dim in mod["main"].params[1].checked_type.shape)
    data = ((np.arange(np.prod(data_shape)) % 17) - 8).reshape(data_shape)
    weight = ((np.arange(np.prod(weight_shape)) % 5) - 2).reshape(weight_shape)
    runtime = graph_executor.GraphModule(factory["default"](tvm.cpu(0)))
    runtime.set_input("data", data.astype(env.inp_dtype))
    runtime.set_input("weight", weight.astype(env.wgt_dtype))
    runtime.run()
    output = runtime.get_output(0).numpy()

    assert output.shape == (env.BATCH, 8, 8, env.BLOCK_OUT)
    assert output.dtype == np.dtype(env.out_dtype)


def test_missing_fsim_reports_setup_command(monkeypatch):
    monkeypatch.setattr(simulator, "enabled", lambda: False)

    with pytest.raises(RuntimeError, match="build_vta_lib.sh --target libvta_fsim"):
        _require_fsim()


def test_loaded_artifact_must_implement_expected_symbol():
    data = tvm.te.placeholder((1,), name="data")
    output = tvm.te.compute((1,), lambda index: data[index], name="output")
    unrelated = tvm.build(
        tvm.te.create_schedule(output.op),
        [data, output],
        target="llvm",
        name="unrelated",
    )

    with pytest.raises(RuntimeError, match="loaded VTA artifact.*tvmgen_missing_vta_main_0"):
        _require_runtime_symbol(unrelated, "tvmgen_missing_vta_main_0")
