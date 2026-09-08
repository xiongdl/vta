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


def _simulator_setup(env):
    if env.TARGET == "sim":
        return (
            "libvta_fsim",
            "vta.simulator.profiler_clear",
            "vta.simulator.profiler_status",
            "./scripts/build_vta_lib.sh --target libvta_fsim",
        )
    if env.TARGET == "tsim":
        return (
            "libvta_tsim + libvta_hw",
            "vta.tsim.profiler_clear",
            "vta.tsim.profiler_status",
            "./scripts/build_vta_lib.sh --target libvta_hw",
        )
    raise RuntimeError(
        "VTA BYOC runtime validation requires sim or tsim, got " f"{env.TARGET}"
    )


def _remote_simulator_stats(remote, status_name):
    status = remote.get_function(status_name)
    return json.loads(status())


def _require_simulator(env):
    library, clear_name, status_name, build_command = _simulator_setup(env)
    if (
        tvm.get_global_func(clear_name, allow_missing=True) is None
        or tvm.get_global_func(status_name, allow_missing=True) is None
    ):
        raise RuntimeError(
            f"VTA {library} is unavailable; run {build_command}"
        )
    return clear_name, status_name


def _assert_accelerator_activity(env, runtime_stats):
    if env.TARGET == "sim":
        assert runtime_stats["gemm_counter"] > 0
        assert runtime_stats["wgt_load_nbytes"] > 0
        assert runtime_stats["out_store_nbytes"] > 0
        return
    if env.TARGET == "tsim":
        assert runtime_stats["cycle_count"] > 0
        return
    raise RuntimeError(
        "VTA BYOC runtime validation requires sim or tsim, got " f"{env.TARGET}"
    )


def _require_runtime_symbol(module, symbol):
    if not module.implements_function(symbol, True):
        raise RuntimeError(f"loaded VTA artifact does not implement {symbol}")


def _test_exported_approved_graph_executes(bias_kind):
    env = vta.get_env()
    clear_name, status_name = _require_simulator(env)
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

    remote.get_function(clear_name)()
    runtime = graph_executor.create(factory.get_graph_json(), loaded, remote.ext_dev(0))
    runtime.set_input("data", input_data)
    assert all(counter == 0 for counter in _remote_simulator_stats(remote, status_name).values())
    runtime.run()
    actual = runtime.get_output(0).numpy()
    runtime_stats = _remote_simulator_stats(remote, status_name)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    _assert_accelerator_activity(env, runtime_stats)


def test_exported_no_bias_graph_executes_on_simulator():
    _test_exported_approved_graph_executes(None)


def test_exported_bias_add_graph_executes_on_simulator():
    _test_exported_approved_graph_executes("bias_add")


def test_exported_broadcast_add_graph_executes_on_simulator():
    _test_exported_approved_graph_executes("add")


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


def test_missing_simulator_reports_setup_command(monkeypatch):
    env = vta.get_env()
    _, clear_name, _, build_command = _simulator_setup(env)
    monkeypatch.setattr(tvm, "get_global_func", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match=build_command):
        _require_simulator(env)


def test_unsupported_simulator_target_reports_target_name(monkeypatch):
    class UnsupportedEnvironment:
        TARGET = "pynq"

    with pytest.raises(RuntimeError, match="requires sim or tsim, got pynq"):
        _simulator_setup(UnsupportedEnvironment())


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
