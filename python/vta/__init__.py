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

    missing_hooks = [
        name
        for name in ("RelayToTIR", "TIRToRuntime")
        if target.get_kind_attr(name) is None
    ]
    if target.get_target_device_type() != tvm.runtime.Device.kDLExtDev or missing_hooks:
        detail = "wrong device type"
        if missing_hooks:
            detail = f"missing target hooks: {', '.join(missing_hooks)}"
        raise ImportError(
            f"VTA compiler extension {extension_path} has an invalid TargetKind 'vta' "
            f"registration ({detail}).\n"
            f"Rebuild it with: {_COMPILER_EXTENSION_BUILD_COMMAND}"
        )

    _COMPILER_EXTENSION_HANDLE = extension


# do not from tvm import topi when running vta.exec.rpc_server
# in lib tvm runtime only mode
if not tvm._ffi.base._RUNTIME_ONLY:
    _load_compiler_extension()
    from . import top
    from .build_module import build_config, lower, build
    from .relay.backend import register_byoc
