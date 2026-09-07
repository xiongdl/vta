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
        assert factory.get_lib().implements_function(symbol, True)
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
        assert factory.get_lib().implements_function(symbol, True)
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
