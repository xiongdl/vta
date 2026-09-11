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

"""VTA Package is a TVM backend extension to support VTA hardware.

Besides the compiler toolchain, it also includes utility functions to
configure the hardware environment and access remote device through RPC.
"""
import sys
import tvm._ffi.base

from .autotvm import module_loader
from .bitstream import get_bitstream_path, download_bitstream
from .environment import get_env, Environment
from .rpc_client import reconfig_runtime, program_fpga

__version__ = "0.1.0"


if "_COMPILER_EXTENSION_HANDLE" not in globals():
    _COMPILER_EXTENSION_HANDLE = None


def _load_compiler_extension():
    """Load and validate the native compiler extension once per process."""
    global _COMPILER_EXTENSION_HANDLE
    if _COMPILER_EXTENSION_HANDLE is not None:
        return

    import ctypes

    from .libinfo import _COMPILER_EXTENSION_BUILD_COMMAND, _find_compiler_extension

    extension_path = _find_compiler_extension()
    try:
        extension = ctypes.CDLL(
            extension_path, mode=getattr(ctypes, "RTLD_GLOBAL", 0)
        )
    except OSError as err:
        raise ImportError(
            f"Unable to load VTA compiler extension {extension_path}: {err}\n"
            f"Rebuild it with: {_COMPILER_EXTENSION_BUILD_COMMAND}"
        ) from err

    try:
        target = tvm.target.Target("vta")
    except ValueError as err:
        raise ImportError(
            f"VTA compiler extension {extension_path} did not register TargetKind 'vta'.\n"
            f"Rebuild it with: {_COMPILER_EXTENSION_BUILD_COMMAND}"
        ) from err

    if target.get_target_device_type() != tvm.runtime.Device.kDLExtDev:
        raise ImportError(
            f"VTA compiler extension {extension_path} has an invalid TargetKind 'vta' "
            "registration (wrong device type).\n"
            f"Rebuild it with: {_COMPILER_EXTENSION_BUILD_COMMAND}"
        )

    required_hooks = (
        ("RelayToTIR", tvm.transform.ModulePass, "tvm.transform.ModulePass"),
        ("TIRToRuntime", tvm.runtime.PackedFunc, "tvm.runtime.PackedFunc"),
    )
    for hook_name, expected_type, expected_type_name in required_hooks:
        hook = target.get_kind_attr(hook_name)
        if hook is None:
            raise ImportError(
                f"VTA compiler extension {extension_path} is missing required "
                f"{hook_name} hook (expected {expected_type_name}).\n"
                f"Rebuild it with: {_COMPILER_EXTENSION_BUILD_COMMAND}"
            )
        if not isinstance(hook, expected_type):
            raise ImportError(
                f"VTA compiler extension {extension_path} {hook_name} hook has "
                f"incompatible type {type(hook).__name__}; expected "
                f"{expected_type_name}.\n"
                f"Rebuild it with: {_COMPILER_EXTENSION_BUILD_COMMAND}"
            )

    _COMPILER_EXTENSION_HANDLE = extension


# do not from tvm import topi when running vta.exec.rpc_server
# in lib tvm runtime only mode
if not tvm._ffi.base._RUNTIME_ONLY:
    _load_compiler_extension()
    from . import relay, top
    from .build_module import build_config, lower, build
    # Register the private Python bridge consumed by the native RelayToTIR hook.
    from .relay import transform as _relay_transform
