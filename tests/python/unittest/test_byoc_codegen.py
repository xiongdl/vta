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

import copy
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import tvm
import vta
import vta.relay.transform as vta_transform
from tvm import relay, te

from byoc_utils import make_qnn_conv2d_module
from vta.relay import partition_for_vta
from vta.relay.backend import _compile_vta_function
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


def _run_isolated_python(source):
    test_dir = str(Path(__file__).resolve().parent)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {test_dir!r})\n{textwrap.dedent(source)}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _tir_to_runtime_hook():
    hook = tvm.target.Target("vta").get_kind_attr("TIRToRuntime")
    assert hook is not None
    return hook


def _vta_tir_module(function_count=1):
    primfunc = lower_vta_function(_partitioned_function())
    functions = {}
    symbols = []
    for index in range(function_count):
        symbol = "tvmgen_native_vta_{}".format(index)
        functions[symbol] = primfunc.with_attr("global_symbol", symbol)
        symbols.append(symbol)
    target = tvm.target.Target("vta", host=vta.get_env().target_host)
    return tvm.IRModule(functions), target, symbols


def _llvm_function_body(llvm_source, symbol):
    definition = re.search(
        r'^define\b[^\n]*@"?{}"?\('.format(re.escape(symbol)),
        llvm_source,
        flags=re.MULTILINE,
    )
    assert definition is not None, symbol
    body_end = llvm_source.find("\n}", definition.end())
    assert body_end != -1, symbol
    return llvm_source[definition.start() : body_end]


@pytest.mark.parametrize("function_count", [1, 3])
def test_tir_to_runtime_returns_one_standard_llvm_module_with_every_symbol(function_count):
    mod, target, symbols = _vta_tir_module(function_count)

    runtime_module = _tir_to_runtime_hook()(mod, target)

    assert isinstance(runtime_module, tvm.runtime.Module)
    assert runtime_module.type_key == "llvm"
    assert runtime_module.handle.value is not None
    assert runtime_module.imported_modules == []
    llvm_source = runtime_module.get_source("ll")
    for symbol in symbols:
        assert runtime_module.implements_function(symbol, False)
        definitions = re.findall(
            r'^define\b[^\n]*@"?{}"?\('.format(re.escape(symbol)),
            llvm_source,
            flags=re.MULTILINE,
        )
        assert len(definitions) == 1, symbol


def test_tir_to_runtime_flattens_external_buffers_before_llvm_codegen():
    mod, target, symbols = _vta_tir_module()
    original_buffers = list(mod[symbols[0]].buffer_map.values())
    assert any(len(buffer.shape) > 1 for buffer in original_buffers)

    runtime_module = _tir_to_runtime_hook()(mod, target)

    assert runtime_module.implements_function(symbols[0], False)


def test_tir_to_runtime_injects_fingerprint_check_before_every_entry_activity():
    mod, target, symbols = _vta_tir_module(3)

    runtime_module = _tir_to_runtime_hook()(mod, target)
    llvm_source = runtime_module.get_source("ll")

    for symbol in symbols:
        body = _llvm_function_body(llvm_source, symbol)
        check_index = body.find("VTACheckConfig")
        assert check_index >= 0, symbol
        activity_indices = [
            body.find(runtime_symbol)
            for runtime_symbol in (
                "VTABufferAlloc",
                "VTATLSCommandHandle",
                "VTABufferCPUPtr",
                "VTALoadBuffer2D",
                "VTAStoreBuffer2D",
                "VTAUopPush",
                "VTAPushGEMMOp",
                "VTAPushALUOp",
                "VTADepPush",
                "VTADepPop",
                "VTASetDebugMode",
                "VTASynchronize",
            )
            if body.find(runtime_symbol) >= 0
        ]
        assert activity_indices, symbol
        assert check_index < min(activity_indices), symbol


def test_tir_to_runtime_does_not_recurse_through_public_tvm_build(monkeypatch):
    mod, target, symbols = _vta_tir_module()

    def unexpected_build(*args, **kwargs):
        raise AssertionError("native TIRToRuntime must not call public tvm.build")

    monkeypatch.setattr(tvm, "build", unexpected_build)

    runtime_module = _tir_to_runtime_hook()(mod, target)

    assert runtime_module.implements_function(symbols[0], False)


def test_tir_to_runtime_validates_every_function_before_codegen():
    mod, target, symbols = _vta_tir_module(2)
    malformed_symbol = symbols[1]
    malformed = mod[malformed_symbol].without_attr("global_symbol")
    mod.update_func(mod.get_global_var(malformed_symbol), malformed)
    before = tvm.ir.save_json(mod)
    llvm_builder = tvm.get_global_func("target.build.llvm")
    codegen_calls = []

    def tracking_llvm_builder(*args):
        codegen_calls.append(args)
        return llvm_builder(*args)

    tvm.register_func("target.build.llvm", tracking_llvm_builder, override=True)

    try:
        with pytest.raises(tvm.error.TVMError, match=malformed_symbol):
            _tir_to_runtime_hook()(mod, target)
    finally:
        tvm.register_func("target.build.llvm", llvm_builder, override=True)

    assert codegen_calls == []
    assert tvm.ir.save_json(mod) == before


def test_tir_to_runtime_rejects_duplicate_symbols():
    mod, target, symbols = _vta_tir_module(2)
    duplicate = mod[symbols[1]].with_attr("global_symbol", symbols[0])
    mod.update_func(mod.get_global_var(symbols[1]), duplicate)

    with pytest.raises(tvm.error.TVMError, match="duplicate.*{}".format(symbols[0])):
        _tir_to_runtime_hook()(mod, target)


def test_tir_to_runtime_requires_global_var_and_symbol_to_match():
    mod, target, symbols = _vta_tir_module()
    symbol = symbols[0]
    mismatched = mod[symbol].with_attr("global_symbol", "tvmgen_wrong_vta_symbol")
    mod.update_func(mod.get_global_var(symbol), mismatched)

    with pytest.raises(tvm.error.TVMError, match="{}.*global_symbol".format(symbol)):
        _tir_to_runtime_hook()(mod, target)


def test_tir_to_runtime_rejects_non_primfunc_and_empty_modules():
    _, target, _ = _vta_tir_module()
    relay_mod = tvm.IRModule.from_expr(relay.Function([], relay.const(0)))

    with pytest.raises(tvm.error.TVMError, match="PrimFunc"):
        _tir_to_runtime_hook()(relay_mod, target)
    with pytest.raises(tvm.error.TVMError, match="empty"):
        _tir_to_runtime_hook()(tvm.IRModule(), target)


def test_tir_to_runtime_rejects_wrong_target_and_missing_llvm_host():
    mod, target, symbols = _vta_tir_module()

    with pytest.raises(tvm.error.TVMError, match="vta target"):
        _tir_to_runtime_hook()(mod, tvm.target.Target("llvm"))
    with pytest.raises(tvm.error.TVMError, match="LLVM host"):
        _tir_to_runtime_hook()(mod, tvm.target.Target("vta"))

    symbol = symbols[0]
    wrong_function_target = mod[symbol].with_attr("target", tvm.target.Target("llvm"))
    mod.update_func(mod.get_global_var(symbol), wrong_function_target)
    with pytest.raises(tvm.error.TVMError, match="{}.*target".format(symbol)):
        _tir_to_runtime_hook()(mod, target)


def test_tir_to_runtime_rejects_malformed_runtime_calls():
    mod, target, symbols = _vta_tir_module()
    symbol = symbols[0]
    malformed = mod[symbol].with_body(
        tvm.tir.Evaluate(tvm.tir.call_extern("int32", "VTAUnknownRuntimeCall"))
    )
    mod.update_func(mod.get_global_var(symbol), malformed)

    with pytest.raises(tvm.error.TVMError, match="{}.*runtime call".format(symbol)):
        _tir_to_runtime_hook()(mod, target)


def test_compile_and_export_do_not_load_or_require_fsim():
    result = _run_isolated_python(
        """
        import os
        import sys
        import tempfile

        import tvm
        import vta
        from tvm import relay

        from byoc_utils import make_qnn_conv2d_module
        from vta.relay import partition_for_vta

        assert "vta.testing.simulator" not in sys.modules
        assert tvm.get_global_func("vta.simulator.profiler_status", True) is None
        assert tvm.get_global_func("relay.ext.vta", True) is None

        env = vta.get_env()
        partitioned = partition_for_vta(
            make_qnn_conv2d_module(env),
            mod_name="compile_without_fsim",
        )
        symbol = next(
            function.attrs.get_str("global_symbol")
            for function in partitioned.functions.values()
            if isinstance(function, relay.Function)
            and function.attrs is not None
            and "Compiler" in function.attrs
        )
        with vta.build_config():
            factory = relay.build(
                partitioned,
                target=tvm.target.Target("vta", host=env.target_host),
            )
        assert factory.get_lib().get_function(symbol, True) is not None

        with tempfile.TemporaryDirectory() as artifact_dir:
            artifact_path = os.path.join(artifact_dir, "compile_without_fsim.tar")
            factory.export_library(artifact_path)
            assert os.path.isfile(artifact_path)
        assert "vta.testing.simulator" not in sys.modules
        assert tvm.get_global_func("vta.simulator.profiler_status", True) is None
        assert tvm.get_global_func("relay.ext.vta", True) is None
        """
    )

    assert result.returncode == 0, result.stderr


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


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_compile_vta_function_returns_exact_symbol(bias_kind):
    external = _partitioned_function(bias_kind)
    symbol = external.attrs.get_str("global_symbol")

    module = _compile_vta_function(external)

    assert isinstance(module, tvm.runtime.Module)
    assert module.implements_function(symbol, True)


def test_compile_vta_function_accepts_tvms_compiler_stripped_input():
    external = _partitioned_function().without_attr("Compiler")
    symbol = external.attrs.get_str("global_symbol")

    module = _compile_vta_function(external)

    assert module.implements_function(symbol, True)


def test_compile_vta_function_rejects_invalid_input_before_build(monkeypatch):
    build_called = False

    def unexpected_build(*args, **kwargs):
        nonlocal build_called
        build_called = True
        raise AssertionError("tvm.build must not run for invalid input")

    monkeypatch.setattr(tvm, "build", unexpected_build)

    with pytest.raises(TypeError, match="func must be a tvm.relay.Function"):
        _compile_vta_function(None)

    assert not build_called


def test_compile_vta_function_rejects_malformed_function_before_build(monkeypatch):
    external = _partitioned_function().without_attr("Primitive")
    build_called = False

    def unexpected_build(*args, **kwargs):
        nonlocal build_called
        build_called = True
        raise AssertionError("tvm.build must not run for malformed input")

    monkeypatch.setattr(tvm, "build", unexpected_build)

    with pytest.raises(ValueError, match="Primitive=1"):
        _compile_vta_function(external)

    assert not build_called


def test_register_byoc_is_explicit_and_idempotent_in_isolated_process():
    result = _run_isolated_python(
        """
        import tvm
        import vta

        assert tvm.get_global_func("relay.ext.vta", True) is None
        vta.register_byoc()
        first = tvm.get_global_func("relay.ext.vta")
        vta.register_byoc()
        second = tvm.get_global_func("relay.ext.vta")
        assert first.handle.value == second.handle.value
        """
    )

    assert result.returncode == 0, result.stderr


def test_register_byoc_rejects_foreign_callback_in_isolated_process():
    result = _run_isolated_python(
        """
        import tvm
        import vta

        foreign = tvm.register_func("relay.ext.vta", lambda func: None)
        try:
            vta.register_byoc()
        except RuntimeError as err:
            assert "relay.ext.vta is already registered" in str(err)
        else:
            raise AssertionError("foreign callback was silently replaced")
        current = tvm.get_global_func("relay.ext.vta")
        assert current.handle.value == foreign.handle.value
        """
    )

    assert result.returncode == 0, result.stderr


def test_registered_vta_compiler_integrates_with_relay_build():
    result = _run_isolated_python(
        """
        import tvm
        import vta
        from tvm import relay

        from byoc_utils import make_qnn_conv2d_module
        from vta.relay import partition_for_vta

        env = vta.get_env()
        partitioned = partition_for_vta(
            make_qnn_conv2d_module(env),
            mod_name="integration",
        )
        external = next(
            function
            for function in partitioned.functions.values()
            if isinstance(function, relay.Function)
            and function.attrs is not None
            and "Compiler" in function.attrs
        )
        symbol = external.attrs.get_str("global_symbol")
        host_ir = partitioned["main"].astext(show_meta_data=False)
        assert "abs" in host_ir
        assert "transpose" in host_ir

        vta.register_byoc()
        target = tvm.target.Target(env.target, host=env.target_host)
        factory = relay.build(partitioned, target=target)
        assert factory.get_lib().get_function(symbol, True) is not None
        """
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_registered_vta_compiler_builds_every_approved_variant(bias_kind):
    result = _run_isolated_python(
        f"""
        import tvm
        import vta
        from tvm import relay

        from byoc_utils import make_qnn_conv2d_module
        from vta.relay import partition_for_vta

        env = vta.get_env()
        partitioned = partition_for_vta(
            make_qnn_conv2d_module(env, bias_kind={bias_kind!r}),
            mod_name="variant",
        )
        symbol = next(
            function.attrs.get_str("global_symbol")
            for function in partitioned.functions.values()
            if isinstance(function, relay.Function)
            and function.attrs is not None
            and "Compiler" in function.attrs
        )
        vta.register_byoc()
        factory = relay.build(
            partitioned,
            target=tvm.target.Target(env.target, host=env.target_host),
        )
        assert factory.get_lib().get_function(symbol, True) is not None
        """
    )

    assert result.returncode == 0, result.stderr


def test_compile_vta_function_rejects_active_configuration_mismatch(monkeypatch):
    external = _partitioned_function()
    incompatible_env = copy.copy(vta.get_env())
    incompatible_env.BLOCK_OUT = 32
    build_called = False

    def unexpected_build(*args, **kwargs):
        nonlocal build_called
        build_called = True
        raise AssertionError("tvm.build must not run for an incompatible configuration")

    monkeypatch.setattr(vta_transform, "get_env", lambda: incompatible_env)
    monkeypatch.setattr(tvm, "build", unexpected_build)

    with pytest.raises(ValueError, match="active VTA configuration"):
        _compile_vta_function(external)

    assert not build_called


def test_compile_vta_function_rejects_undefined_runtime_module(monkeypatch):
    external = _partitioned_function()
    symbol = external.attrs.get_str("global_symbol")
    monkeypatch.setattr(tvm, "build", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match=f"no runtime module for {symbol}"):
        _compile_vta_function(external)


def test_compile_vta_function_rejects_runtime_module_without_symbol(monkeypatch):
    data = te.placeholder((1,), name="data")
    output = te.compute((1,), lambda i: data[i], name="output")
    unrelated_module = tvm.build(
        te.create_schedule(output.op),
        [data, output],
        target="llvm",
        name="unrelated",
    )
    external = _partitioned_function()
    symbol = external.attrs.get_str("global_symbol")
    monkeypatch.setattr(tvm, "build", lambda *args, **kwargs: unrelated_module)

    with pytest.raises(RuntimeError, match=f"does not implement {symbol}"):
        _compile_vta_function(external)


def test_near_miss_build_does_not_invoke_vta_codegen():
    result = _run_isolated_python(
        """
        import tvm
        import vta
        from tvm import relay

        from byoc_utils import make_qnn_conv2d_near_miss_module
        from vta.relay import partition_for_vta

        calls = []
        tvm.register_func("relay.ext.vta", lambda func: calls.append(func))
        partitioned = partition_for_vta(make_qnn_conv2d_near_miss_module(vta.get_env())[0])
        relay.build(partitioned, target="llvm")
        assert calls == []
        """
    )

    assert result.returncode == 0, result.stderr
