"""Pass-only Relay partition pipeline for VTA."""

import tvm
from tvm import relay
from tvm.relay.build_module import bind_params_by_name

from ..config import VTAConfig
from .patterns import pattern_table


def partition_for_vta(mod, params=None, config=None, mod_name="default"):
    """Discover supported VTA regions and produce Compiler=\"vta\" functions.

    The input graph remains in its standard logical layout. Packing and channel
    padding intentionally belong to the later VTA Relay-to-TIR pipeline.
    """
    config = config or VTAConfig.from_json()
    if params:
        mod = tvm.IRModule(mod.functions, mod.type_definitions, mod.attrs)
        mod["main"] = bind_params_by_name(mod["main"], params)

    pipeline = tvm.transform.Sequential(
        [
            relay.transform.InferType(),
            relay.transform.FoldConstant(),
            relay.transform.MergeComposite(pattern_table(config)),
            relay.transform.AnnotateTarget("vta"),
            relay.transform.MergeCompilerRegions(),
            relay.transform.PartitionGraph(mod_name=mod_name),
            relay.transform.InferType(),
        ],
        name="VTA Relay partition pipeline",
    )
    return pipeline(mod)

