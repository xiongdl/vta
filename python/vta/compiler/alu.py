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


def _compile_add_shapes(logical_shape, packed_shape, name, input_dtype):
    env = get_env()
    lhs = te.placeholder(packed_shape, dtype=input_dtype, name="lhs")
    rhs = te.placeholder(packed_shape, dtype=input_dtype, name="rhs")
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
        lower(schedule, args, simple_mode=True), logical_shape, packed_shape,
    )


def compile_add(function, name):
    """Compile a logical Relay residual add with a flat tiled ABI."""
    env = get_env()
    shape = tuple(int(x) for x in function.params[0].checked_type.shape)
    elements = int(np.prod(shape))
    lanes = env.BATCH * env.BLOCK_OUT
    padded = ((elements + lanes - 1) // lanes) * lanes
    return _compile_add_shapes(
        shape, (1, padded // lanes, env.BATCH, env.BLOCK_OUT), name, env.acc_dtype
    )


def compile_packed_add(logical_shape, packed_shape, name):
    """Compile add directly over an existing VTA packed Conv2d boundary."""
    env = get_env()
    elements = int(np.prod(packed_shape))
    canonical = (1, elements // (env.BATCH * env.BLOCK_OUT), env.BATCH, env.BLOCK_OUT)
    return _compile_add_shapes(tuple(logical_shape), canonical, name, env.acc_dtype)
