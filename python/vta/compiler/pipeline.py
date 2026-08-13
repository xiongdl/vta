"""Top-level compiler orchestration owned by tvm-vta."""

from ..relay import partition_for_vta
from tvm import relay

from .conv2d import compile_qnn_conv2d
from .dense import _find_call, compile_qnn_dense
from .executor import ExecutionPlan


def _compile_regions(partitioned):
    artifacts = {}
    for global_var, function in partitioned.functions.items():
        if not (
            isinstance(function, relay.Function)
            and function.attrs
            and function.attrs.get("Compiler") == "vta"
        ):
            continue
        name = global_var.name_hint
        if _find_call(function.body, "qnn.conv2d") is not None:
            artifacts[name] = compile_qnn_conv2d(function, name=name)
        elif _find_call(function.body, "qnn.dense") is not None:
            artifacts[name] = compile_qnn_dense(function, name=name)
        else:
            raise ValueError(f"VTA region {name} contains no supported core operator")
    return artifacts


def compile(mod, params=None, config=None):
    """Partition a Relay module and compile its single supported VTA region."""
    partitioned = partition_for_vta(mod, params=params, config=config)
    artifacts = _compile_regions(partitioned)
    if not artifacts:
        raise ValueError("Expected at least one VTA region, found none")
    if len(artifacts) == 1:
        return next(iter(artifacts.values()))
    return ExecutionPlan(partitioned, artifacts)


def compile_plan(mod, params=None, config=None):
    """Always return a mixed CPU/VTA execution plan, including for one region."""
    partitioned = partition_for_vta(mod, params=params, config=config)
    artifacts = _compile_regions(partitioned)
    if not artifacts:
        raise ValueError("Expected at least one VTA region, found none")
    return ExecutionPlan(partitioned, artifacts)
