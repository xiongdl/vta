"""Registration boundary for the out-of-tree VTA compiler."""

import threading

from .dense import DenseArtifact, compile_partitioned_dense, compile_qnn_dense

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


__all__ = ["DenseArtifact", "compile_partitioned_dense", "compile_qnn_dense", "register_compiler"]
