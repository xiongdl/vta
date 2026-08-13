"""Top-level compiler orchestration owned by tvm-vta."""

from ..relay import partition_for_vta
from tvm import relay

from .conv2d import compile_qnn_conv2d
from .dense import _find_call, compile_qnn_dense


def compile(mod, params=None, config=None):
    """Partition a Relay module and compile its single supported VTA region."""
    partitioned = partition_for_vta(mod, params=params, config=config)
    candidates = [
        (global_var.name_hint, function)
        for global_var, function in partitioned.functions.items()
        if isinstance(function, relay.Function)
        and function.attrs
        and function.attrs.get("Compiler") == "vta"
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one VTA region, found {len(candidates)}")
    name, function = candidates[0]
    if _find_call(function.body, "qnn.conv2d") is not None:
        return compile_qnn_conv2d(function, name=name)
    if _find_call(function.body, "qnn.dense") is not None:
        return compile_qnn_dense(function, name=name)
    raise ValueError("The VTA region contains no supported core operator")
