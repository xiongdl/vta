"""End-to-end TE -> VTA TIR -> standalone FSIM test."""

import numpy as np
import tvm
from tvm import rpc, te
from tvm.contrib import utils

import vta
from vta.testing import simulator


def test_fsim_vector_add():
    env = vta.get_env()
    assert env.TARGET == "sim"
    assert simulator.enabled()

    shape = (1, 4, env.BATCH, env.BLOCK_OUT)
    lhs = te.placeholder(shape, name="lhs", dtype=env.acc_dtype)
    rhs = te.placeholder(shape, name="rhs", dtype=env.acc_dtype)
    lhs_buf = te.compute(shape, lambda *idx: lhs(*idx), name="lhs_buf")
    rhs_buf = te.compute(shape, lambda *idx: rhs(*idx), name="rhs_buf")
    result_buf = te.compute(shape, lambda *idx: lhs_buf(*idx) + rhs_buf(*idx), name="result_buf")
    result = te.compute(shape, lambda *idx: result_buf(*idx).astype(env.inp_dtype), name="result")

    schedule = te.create_schedule(result.op)
    for tensor in (lhs_buf, rhs_buf, result_buf):
        schedule[tensor].set_scope(env.acc_scope)
    schedule[lhs_buf].pragma(schedule[lhs_buf].op.axis[0], env.dma_copy)
    schedule[rhs_buf].pragma(schedule[rhs_buf].op.axis[0], env.dma_copy)
    schedule[result_buf].pragma(schedule[result_buf].op.axis[0], env.alu)
    schedule[result].pragma(schedule[result].op.axis[0], env.dma_copy)

    lowered = str(vta.lower(schedule, [lhs, rhs, result], simple_mode=True))
    assert "tir.vta.uop_push" in lowered
    module = vta.build(schedule, [lhs, rhs, result],
                       tvm.target.Target("ext_dev", host=env.target_host),
                       name="standalone_vadd")

    remote = rpc.LocalSession()
    temp = utils.tempdir()
    module.save(temp.relpath("standalone_vadd.o"))
    remote.upload(temp.relpath("standalone_vadd.o"))
    module = remote.load_module("standalone_vadd.o")
    device = remote.ext_dev(0)
    lhs_np = np.random.randint(-128, 128, size=shape).astype("int32")
    rhs_np = np.random.randint(-128, 128, size=shape).astype("int32")
    lhs_nd = tvm.nd.array(lhs_np, device)
    rhs_nd = tvm.nd.array(rhs_np, device)
    result_nd = tvm.nd.empty(shape, env.inp_dtype, device)
    module(lhs_nd, rhs_nd, result_nd)
    np.testing.assert_equal(result_nd.numpy(), (lhs_np + rhs_np).astype("int8"))
