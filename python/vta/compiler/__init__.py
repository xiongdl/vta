"""Registration boundary for the out-of-tree VTA compiler."""

import threading

from .dense import DenseArtifact, compile_partitioned_dense, compile_qnn_dense
from .conv2d import Conv2DArtifact, compile_qnn_conv2d
from .alu import AddArtifact, compile_add
from .pipeline import compile, compile_plan
from .executor import ExecutionPlan, run_artifact
from .codegen import build_graph, register_external_codegen

_LOCK = threading.Lock()
_REGISTERED = False


def register_compiler():
    """Register compiler hooks without importing the legacy in-tree package.

    Relay-to-TIR registration is deliberately kept behind this explicit API so
    importing :mod:`vta` has no process-global TVM side effects.
    """
    global _REGISTERED
    with _LOCK:
        if _REGISTERED:
            return False
        # The first implementation milestone provides partitioning. The
        # Relay-to-TIR callback is added here once its generated PrimFuncs are
        # equivalent to the existing VTA lowering baseline.
        _REGISTERED = True
        return True


__all__ = ["AddArtifact", "Conv2DArtifact", "DenseArtifact", "ExecutionPlan", "build_graph", "compile",
           "compile_add",
           "compile_plan",
           "compile_partitioned_dense",
           "compile_qnn_conv2d", "compile_qnn_dense",
           "register_compiler", "register_external_codegen", "run_artifact"]
