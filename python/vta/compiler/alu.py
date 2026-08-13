"""VTA ALU compiler for residual tensor addition."""

from dataclasses import dataclass
import numpy as np
import tvm
from tvm import te

from ..build_module import build, lower
from ..environment import get_env


@dataclass
class AddArtifact:
    module: tvm.runtime.Module
    lowered: tvm.IRModule
    logical_shape: tuple
    packed_shape: tuple


def compile_add(function, name):
    env = get_env()
    shape = tuple(int(x) for x in function.params[0].checked_type.shape)
    elements = int(np.prod(shape))
    lanes = env.BATCH * env.BLOCK_OUT
    padded = ((elements + lanes - 1) // lanes) * lanes
    # Retain the canonical four-dimensional VTA ALU layout.  The instruction
    # injector uses the outer two axes when deriving SRAM strides.
    packed_shape = (1, padded // lanes, env.BATCH, env.BLOCK_OUT)
    lhs = te.placeholder(packed_shape, dtype=env.acc_dtype, name="lhs")
    rhs = te.placeholder(packed_shape, dtype=env.acc_dtype, name="rhs")
    lhs_buf = te.compute(packed_shape, lambda *i: lhs(*i), "lhs_buf")
    rhs_buf = te.compute(packed_shape, lambda *i: rhs(*i), "rhs_buf")
    added = te.compute(packed_shape, lambda *i: lhs_buf(*i) + rhs_buf(*i), "added")
    output = te.compute(packed_shape, lambda *i: added(*i).astype(env.out_dtype), "output")
    schedule = te.create_schedule(output.op)
    for stage in (lhs_buf, rhs_buf, added):
        schedule[stage].set_scope(env.acc_scope)
    for stage in (lhs_buf, rhs_buf): schedule[stage].pragma(schedule[stage].op.axis[0], env.dma_copy)
    schedule[added].pragma(schedule[added].op.axis[0], env.alu)
    schedule[output].pragma(schedule[output].op.axis[0], env.dma_copy)
    args = [lhs, rhs, output]
    return AddArtifact(
        build(schedule, args, tvm.target.Target("ext_dev", host=env.target_host), name=name),
        lower(schedule, args, simple_mode=True), shape, packed_shape,
    )
