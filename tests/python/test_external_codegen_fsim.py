"""Standard Relay GraphExecutor using the out-of-tree VTA external compiler."""

import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor

import vta


def _dense(data, weight):
    dense = relay.qnn.op.dense(
        data,
        relay.const(weight),
        relay.const(0, "int32"), relay.const(0, "int32"),
        relay.const(1.0, "float32"), relay.const(1.0, "float32"),
        units=weight.shape[0], out_dtype="int32",
    )
    return relay.qnn.op.requantize(
        dense,
        relay.const(1.0, "float32"), relay.const(0, "int32"),
        relay.const(1.0, "float32"), relay.const(0, "int32"),
        out_dtype="int8",
    )


def test_graph_executor_cpu_vta_dense():
    env = vta.get_env()
    features = 2 * env.BLOCK_IN
    data_var = relay.var("data", shape=(env.BATCH, features), dtype="int8")
    weight = np.eye(features, dtype="int8")
    vta_value = _dense(data_var, weight)
    result = relay.add(vta_value, relay.const(np.ones((env.BATCH, features), dtype="int8")))
    mod = tvm.IRModule.from_expr(relay.Function([data_var], result))
    factory = vta.build_graph(mod)
    runtime = graph_executor.GraphModule(factory["default"](tvm.cpu()))
    data = np.random.randint(-20, 20, (env.BATCH, features)).astype("int8")
    runtime.set_input("data", data)
    runtime.run()
    np.testing.assert_equal(runtime.get_output(0).numpy(), (data + 1).astype("int8"))
