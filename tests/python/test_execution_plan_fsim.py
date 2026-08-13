"""Mixed CPU and multiple VTA region execution."""

import numpy as np
import tvm
from tvm import relay

import vta


def _dense(data, weight):
    value = relay.qnn.op.dense(
        data,
        relay.const(weight),
        relay.const(0, "int32"),
        relay.const(0, "int32"),
        relay.const(1.0, "float32"),
        relay.const(1.0, "float32"),
        units=weight.shape[0],
        out_dtype="int32",
    )
    return relay.qnn.op.requantize(
        value,
        relay.const(1.0, "float32"),
        relay.const(0, "int32"),
        relay.const(1.0, "float32"),
        relay.const(0, "int32"),
        out_dtype="int8",
    )


def test_two_vta_regions_with_cpu_operator():
    env = vta.get_env()
    # Exercise more than one reduction tile; the single-tile dense schedule has
    # a separate reset-uop lowering edge case covered independently.
    features = 2 * env.BLOCK_IN
    data_var = relay.var("data", shape=(env.BATCH, features), dtype="int8")
    weight1 = np.eye(features, dtype="int8")
    weight2 = np.eye(features, dtype="int8")
    first = _dense(data_var, weight1)
    cpu_value = relay.add(first, relay.const(np.ones((env.BATCH, features), dtype="int8")))
    second = _dense(cpu_value, weight2)
    plan = vta.compile_plan(tvm.IRModule.from_expr(relay.Function([data_var], second)))
    assert len(plan.artifacts) == 2
    data = np.random.randint(-10, 10, (env.BATCH, features)).astype("int8")
    np.testing.assert_equal(plan.run(data=data), (data + 1).astype("int8"))
