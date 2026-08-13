"""Out-of-tree VTA compiler integration.

Importing this package does not enable or build TVM's in-tree VTA runtime.
Compiler registration is explicit through :func:`register_compiler`.
"""

from .config import VTAConfig
from .environment import Environment, get_env
from .relay import partition_for_vta, pattern_table
from .build_module import build, build_config, lower
from .compiler import compile, compile_partitioned_dense, compile_qnn_conv2d, compile_qnn_dense


def register_compiler():
    """Register the Python-side VTA compiler hooks once."""
    from .compiler import register_compiler as _register_compiler

    return _register_compiler()


__all__ = ["Environment", "VTAConfig", "build", "build_config", "compile",
           "compile_partitioned_dense",
           "compile_qnn_conv2d",
           "compile_qnn_dense", "get_env", "lower",
           "partition_for_vta", "pattern_table", "register_compiler"]
