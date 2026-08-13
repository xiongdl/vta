"""Standard Relay GraphExecutor using the out-of-tree VTA external compiler."""

import numpy as np
import time
import pytest
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.topi.testing import conv2d_nchw_python
from tvm.topi.testing import depthwise_conv2d_python_nchw

import vta
from vta.testing import simulator


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


def _conv2d(data, weight, stride=1, padding=0):
    value = relay.qnn.op.conv2d(
        data, relay.const(weight),
        relay.const(0, "int32"), relay.const(0, "int32"),
        relay.const(1.0, "float32"), relay.const(1.0, "float32"),
        strides=(stride, stride), padding=(padding, padding),
        channels=weight.shape[0], kernel_size=weight.shape[2:], out_dtype="int32",
    )
    return relay.qnn.op.requantize(
        value, relay.const(1.0, "float32"), relay.const(0, "int32"),
        relay.const(1.0, "float32"), relay.const(0, "int32"), out_dtype="int8",
    )


def _run(mod, data):
    factory = vta.build_graph(mod)
    runtime = graph_executor.GraphModule(factory["default"](tvm.cpu()))
    runtime.set_input("data", data)
    runtime.run()
    return runtime.get_output(0).numpy()


def _build_runtime(mod):
    factory = vta.build_graph(mod)
    return graph_executor.GraphModule(factory["default"](tvm.cpu()))


def test_graph_executor_cpu_vta_dense():
    env = vta.get_env()
    features = 2 * env.BLOCK_IN
    data_var = relay.var("data", shape=(env.BATCH, features), dtype="int8")
    weight = np.eye(features, dtype="int8")
    vta_value = _dense(data_var, weight)
    result = relay.add(vta_value, relay.const(np.ones((env.BATCH, features), dtype="int8")))
    mod = tvm.IRModule.from_expr(relay.Function([data_var], result))
    data = np.random.randint(-20, 20, (env.BATCH, features)).astype("int8")
    np.testing.assert_equal(_run(mod, data), (data + 1).astype("int8"))


def test_graph_executor_cpu_vta_conv2d_unaligned_stride2():
    data_var = relay.var("data", shape=(1, 3, 7, 7), dtype="int8")
    weight = np.random.randint(-2, 3, (5, 3, 3, 3)).astype("int8")
    value = _conv2d(data_var, weight, stride=2)
    result = relay.add(value, relay.const(np.ones((1, 5, 3, 3), dtype="int8")))
    mod = tvm.IRModule.from_expr(relay.Function([data_var], result))
    data = np.random.randint(-3, 4, (1, 3, 7, 7)).astype("int8")
    expected = conv2d_nchw_python(data.astype("int32"), weight.astype("int32"), 2, 0)
    expected = (np.clip(expected, -128, 127).astype("int8") + 1).astype("int8")
    np.testing.assert_equal(_run(mod, data), expected)


def test_graph_executor_two_vta_conv2d_regions():
    env = vta.get_env()
    data_var = relay.var("data", shape=(1, env.BLOCK_IN, 4, 4), dtype="int8")
    identity = np.zeros((env.BLOCK_OUT, env.BLOCK_IN, 1, 1), dtype="int8")
    for channel in range(min(env.BLOCK_IN, env.BLOCK_OUT)):
        identity[channel, channel, 0, 0] = 1
    first = _conv2d(data_var, identity)
    cpu = relay.add(first, relay.const(np.ones((1, env.BLOCK_OUT, 4, 4), dtype="int8")))
    second = _conv2d(cpu, identity)
    mod = tvm.IRModule.from_expr(relay.Function([data_var], second))
    data = np.random.randint(-10, 10, (1, env.BLOCK_IN, 4, 4)).astype("int8")
    np.testing.assert_equal(_run(mod, data), (data + 1).astype("int8"))


def test_adjacent_conv2d_keeps_packed_boundary_and_reuses_buffers():
    env = vta.get_env()
    data_var = relay.var("data", shape=(1, env.BLOCK_IN, 4, 4), dtype="int8")
    identity = np.zeros((env.BLOCK_OUT, env.BLOCK_IN, 1, 1), dtype="int8")
    for channel in range(env.BLOCK_IN):
        identity[channel, channel, 0, 0] = 1
    result = _conv2d(_conv2d(data_var, identity), identity)
    mod = tvm.IRModule.from_expr(relay.Function([data_var], result))
    partitioned = vta.partition_for_vta(mod)
    regions = [
        function for function in partitioned.functions.values()
        if isinstance(function, relay.Function) and function.attrs
        and function.attrs.get("Compiler") == "vta"
    ]
    assert len(regions) == 1
    runtime = _build_runtime(mod)
    for _ in range(3):
        data = np.random.randint(-10, 10, (1, env.BLOCK_IN, 4, 4)).astype("int8")
        runtime.set_input("data", data)
        runtime.run()
        np.testing.assert_equal(runtime.get_output(0).numpy(), data)


def test_inexact_quantization_stays_on_cpu():
    data = relay.var("data", shape=(1, 16, 4, 4), dtype="int8")
    weight = np.ones((16, 16, 1, 1), dtype="int8")
    value = relay.qnn.op.conv2d(
        data, relay.const(weight), relay.const(1, "int32"), relay.const(0, "int32"),
        relay.const(1.0, "float32"), relay.const(1.0, "float32"),
        channels=16, kernel_size=(1, 1), out_dtype="int32",
    )
    value = relay.qnn.op.requantize(
        value, relay.const(1.0, "float32"), relay.const(0, "int32"),
        relay.const(0.5, "float32"), relay.const(0, "int32"), out_dtype="int8",
    )
    partitioned = vta.partition_for_vta(tvm.IRModule.from_expr(relay.Function([data], value)))
    assert not any(
        isinstance(function, relay.Function) and function.attrs and function.attrs.get("Compiler") == "vta"
        for function in partitioned.functions.values()
    )


def test_relay_017_rejects_per_channel_requantize_scale():
    """Document the upstream TypeRel boundary without changing TVM."""
    data = relay.var("data", shape=(1, 16, 4, 4), dtype="int32")
    value = relay.qnn.op.requantize(
        data, relay.const(np.ones(16, dtype="float32")), relay.const(0, "int32"),
        relay.const(np.ones(16, dtype="float32")), relay.const(0, "int32"),
        axis=1, out_dtype="int8",
    )
    with pytest.raises(tvm.TVMError, match="IsScalarType"):
        relay.transform.InferType()(tvm.IRModule.from_expr(relay.Function([data], value)))


def test_uniform_per_channel_scale_normalization_and_fixed_point_ratio():
    data = relay.var("data", shape=(1, 16, 4, 4), dtype="int32")
    scales = relay.const(np.full(16, 0.25, dtype="float32"))
    value = relay.qnn.op.requantize(
        data, scales, relay.const(0, "int32"), scales, relay.const(0, "int32"),
        axis=1, out_dtype="int8",
    )
    normalized = vta.normalize_qnn_scales(
        tvm.IRModule.from_expr(relay.Function([data], value))
    )
    relay.transform.InferType()(normalized)
    assert vta.fixed_point_ratio(0.5, 0.125) == 2
    assert vta.fixed_point_ratio(0.125, 0.5) == -2
    with pytest.raises(ValueError, match="power-of-two"):
        vta.fixed_point_ratio(0.3, 0.2)


def test_power_of_two_requantize_conv2d_fsim():
    data_var = relay.var("data", shape=(1, 16, 3, 3), dtype="int8")
    weight = np.zeros((16, 16, 1, 1), dtype="int8")
    for channel in range(16):
        weight[channel, channel, 0, 0] = 1
    value = relay.qnn.op.conv2d(
        data_var, relay.const(weight), relay.const(0, "int32"), relay.const(0, "int32"),
        relay.const(1.0, "float32"), relay.const(1.0, "float32"),
        channels=16, kernel_size=(1, 1), out_dtype="int32",
    )
    value = relay.qnn.op.requantize(
        value, relay.const(1.0, "float32"), relay.const(0, "int32"),
        relay.const(0.5, "float32"), relay.const(0, "int32"), out_dtype="int8",
    )
    data = np.random.randint(-20, 21, (1, 16, 3, 3)).astype("int8")
    mod = tvm.IRModule.from_expr(relay.Function([data_var], value))
    np.testing.assert_equal(_run(mod, data), (data.astype("int16") * 2).astype("int8"))


def test_power_of_two_right_shift_and_output_zero_point_fsim():
    data_var = relay.var("data", shape=(1, 16, 3, 3), dtype="int8")
    weight = np.zeros((16, 16, 1, 1), dtype="int8")
    for channel in range(16):
        weight[channel, channel, 0, 0] = 1
    value = relay.qnn.op.conv2d(
        data_var, relay.const(weight), relay.const(0, "int32"), relay.const(0, "int32"),
        relay.const(1.0, "float32"), relay.const(1.0, "float32"),
        channels=16, kernel_size=(1, 1), out_dtype="int32",
    )
    value = relay.qnn.op.requantize(
        value, relay.const(1.0, "float32"), relay.const(0, "int32"),
        relay.const(2.0, "float32"), relay.const(3, "int32"), out_dtype="int8",
    )
    data = np.random.randint(-20, 21, (1, 16, 3, 3)).astype("int8")
    mod = tvm.IRModule.from_expr(relay.Function([data_var], value))
    # TVM's default UPWARD rounding adds half before arithmetic right shift.
    expected = ((data.astype("int32") + 1) >> 1) + 3
    np.testing.assert_equal(_run(mod, data), expected.astype("int8"))


def test_non_power_of_two_requantize_uses_exact_llvm_fallback():
    data = relay.var("data", shape=(1, 16, 2, 2), dtype="int32")
    value = relay.qnn.op.requantize(
        data, relay.const(0.3, "float32"), relay.const(0, "int32"),
        relay.const(0.2, "float32"), relay.const(1, "int32"), out_dtype="int8",
    )
    mod = tvm.IRModule.from_expr(relay.Function([data], value))
    partitioned = vta.partition_for_vta(mod)
    assert not any(
        isinstance(function, relay.Function) and function.attrs
        and function.attrs.get("Compiler") == "vta"
        for function in partitioned.functions.values()
    )
    with tvm.transform.PassContext(opt_level=3):
        factory = relay.build(partitioned, target="llvm")
    runtime = graph_executor.GraphModule(factory["default"](tvm.cpu()))
    values = np.arange(-32, 32, dtype="int32").reshape(1, 16, 2, 2)
    runtime.set_input("data", values)
    runtime.run()
    reference = relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm").evaluate()(
        tvm.nd.array(values)
    ).numpy()
    np.testing.assert_equal(runtime.get_output(0).numpy(), reference)


def test_graph_executor_depthwise_conv2d():
    channels = 16
    data_var = relay.var("data", shape=(1, channels, 5, 5), dtype="int8")
    weight = np.random.randint(-2, 3, (channels, 1, 3, 3)).astype("int8")
    value = relay.qnn.op.conv2d(
        data_var, relay.const(weight), relay.const(0, "int32"), relay.const(0, "int32"),
        relay.const(0.5, "float32"), relay.const(0.25, "float32"),
        padding=(1, 1), groups=channels, channels=channels, kernel_size=(3, 3),
        out_dtype="int32",
    )
    value = relay.qnn.op.requantize(
        value, relay.const(0.125, "float32"), relay.const(0, "int32"),
        relay.const(0.125, "float32"),
        relay.const(0, "int32"), axis=1, out_dtype="int8",
    )
    mod = tvm.IRModule.from_expr(relay.Function([data_var], value))
    data = np.random.randint(-3, 4, (1, channels, 5, 5)).astype("int8")
    expected = depthwise_conv2d_python_nchw(data.astype("int32"), weight.astype("int32"), 1, 1)
    np.testing.assert_equal(_run(mod, data), np.clip(expected, -128, 127).astype("int8"))


def test_graph_executor_group_conv2d():
    channels, groups = 16, 4
    data_var = relay.var("data", shape=(1, channels, 4, 4), dtype="int8")
    weight = np.random.randint(-2, 3, (channels, channels // groups, 1, 1)).astype("int8")
    value = relay.qnn.op.conv2d(
        data_var, relay.const(weight), relay.const(0, "int32"), relay.const(0, "int32"),
        relay.const(1.0, "float32"), relay.const(1.0, "float32"), groups=groups,
        channels=channels, kernel_size=(1, 1), out_dtype="int32",
    )
    value = relay.qnn.op.requantize(
        value, relay.const(1.0, "float32"), relay.const(0, "int32"),
        relay.const(1.0, "float32"), relay.const(0, "int32"), out_dtype="int8",
    )
    data = np.random.randint(-3, 4, (1, channels, 4, 4)).astype("int8")
    full_weight = np.zeros((channels, channels, 1, 1), dtype="int32")
    per_group = channels // groups
    for group in range(groups):
        full_weight[group*per_group:(group+1)*per_group, group*per_group:(group+1)*per_group] = weight[group*per_group:(group+1)*per_group]
    expected = conv2d_nchw_python(data.astype("int32"), full_weight, 1, 0)
    mod = tvm.IRModule.from_expr(relay.Function([data_var], value))
    np.testing.assert_equal(_run(mod, data), np.clip(expected, -128, 127).astype("int8"))


def test_graph_executor_vta_residual_add():
    shape = (1, 16, 4, 4)
    lhs = relay.var("lhs", shape=shape, dtype="int8")
    rhs = relay.var("rhs", shape=shape, dtype="int8")
    mod = tvm.IRModule.from_expr(relay.Function([lhs, rhs], relay.add(lhs, rhs)))
    partitioned = vta.partition_for_vta(mod)
    regions = [
        function for function in partitioned.functions.values()
        if isinstance(function, relay.Function) and function.attrs
        and function.attrs.get("Compiler") == "vta"
    ]
    assert len(regions) == 1
    runtime = _build_runtime(mod)
    lhs_data = np.random.randint(-100, 101, shape).astype("int8")
    rhs_data = np.random.randint(-100, 101, shape).astype("int8")
    runtime.set_input("lhs", lhs_data)
    runtime.set_input("rhs", rhs_data)
    runtime.run()
    expected = (lhs_data.astype("int32") + rhs_data.astype("int32")).astype("int8")
    np.testing.assert_equal(runtime.get_output(0).numpy(), expected)


def test_quantized_resnet_basic_block_fsim():
    """Exercise conv/conv/residual-add as three independently compiled VTA regions."""
    env = vta.get_env()
    shape = (1, env.BLOCK_IN, 4, 4)
    data_var = relay.var("data", shape=shape, dtype="int8")
    identity = np.zeros((env.BLOCK_OUT, env.BLOCK_IN, 1, 1), dtype="int8")
    for channel in range(min(env.BLOCK_IN, env.BLOCK_OUT)):
        identity[channel, channel, 0, 0] = 1
    first = relay.annotation.stop_fusion(relay.clip(_conv2d(data_var, identity), 0, 127))
    second = relay.annotation.stop_fusion(_conv2d(first, identity))
    result = relay.add(second, data_var)
    mod = tvm.IRModule.from_expr(relay.Function([data_var], result))
    partitioned = vta.partition_for_vta(mod)
    regions = [
        function for function in partitioned.functions.values()
        if isinstance(function, relay.Function) and function.attrs
        and function.attrs.get("Compiler") == "vta"
    ]
    assert len(regions) == 3
    runtime = _build_runtime(mod)
    data = np.random.randint(-64, 65, shape).astype("int8")
    runtime.set_input("data", data)
    runtime.run()
    first_expected = np.clip(data.astype("int32"), 0, 127)
    expected = (first_expected + data.astype("int32")).astype("int8")
    np.testing.assert_equal(runtime.get_output(0).numpy(), expected)

    samples = []
    for _ in range(5):
        started = time.perf_counter()
        runtime.run()
        samples.append(time.perf_counter() - started)
    assert np.median(samples) > 0
    print(f"ResNet basic block FSIM median: {np.median(samples) * 1e3:.3f} ms")


def test_fused_resnet_basic_block_keeps_packed_residual():
    env = vta.get_env()
    shape = (1, env.BLOCK_IN, 4, 4)
    data_var = relay.var("data", shape=shape, dtype="int8")
    identity = np.zeros((env.BLOCK_OUT, env.BLOCK_IN, 1, 1), dtype="int8")
    for channel in range(env.BLOCK_IN):
        identity[channel, channel, 0, 0] = 1
    result = relay.add(_conv2d(_conv2d(data_var, identity), identity), data_var)
    mod = tvm.IRModule.from_expr(relay.Function([data_var], result))
    partitioned = vta.partition_for_vta(mod)
    regions = [
        function for function in partitioned.functions.values()
        if isinstance(function, relay.Function) and function.attrs
        and function.attrs.get("Compiler") == "vta"
    ]
    assert len(regions) == 1
    runtime = _build_runtime(mod)
    data = np.random.randint(-64, 64, shape).astype("int8")
    runtime.set_input("data", data)
    runtime.run()
    np.testing.assert_equal(runtime.get_output(0).numpy(), (data + data).astype("int8"))


def test_resnet_stage_with_stride2_downsample():
    """Two blocks including the ResNet stage-transition projection shortcut."""
    env = vta.get_env()
    input_shape = (1, env.BLOCK_IN, 8, 8)
    data_var = relay.var("data", shape=input_shape, dtype="int8")
    out_channels = 2 * env.BLOCK_OUT
    conv1_weight = np.zeros((out_channels, env.BLOCK_IN, 3, 3), dtype="int8")
    projection = np.zeros((out_channels, env.BLOCK_IN, 1, 1), dtype="int8")
    conv2_weight = np.zeros((out_channels, out_channels, 3, 3), dtype="int8")
    identity = np.zeros((out_channels, out_channels, 1, 1), dtype="int8")
    for channel in range(env.BLOCK_IN):
        conv1_weight[channel, channel, 1, 1] = 1
        projection[channel, channel, 0, 0] = 1
    for channel in range(out_channels):
        conv2_weight[channel, channel, 1, 1] = 1
        identity[channel, channel, 0, 0] = 1
    branch = relay.annotation.stop_fusion(
        _conv2d(_conv2d(data_var, conv1_weight, stride=2, padding=1), conv2_weight, padding=1)
    )
    shortcut = relay.annotation.stop_fusion(_conv2d(data_var, projection, stride=2))
    transitioned = relay.annotation.stop_fusion(relay.add(branch, shortcut))
    result = relay.add(_conv2d(_conv2d(transitioned, identity), identity), transitioned)
    mod = tvm.IRModule.from_expr(relay.Function([data_var], result))
    runtime = _build_runtime(mod)
    data = np.random.randint(-20, 21, input_shape).astype("int8")
    runtime.set_input("data", data)
    runtime.run()
    downsampled = data[:, :, ::2, ::2]
    expected = np.zeros((1, out_channels, 4, 4), dtype="int8")
    expected[:, :env.BLOCK_IN] = (downsampled.astype("int16") * 4).astype("int8")
    np.testing.assert_equal(runtime.get_output(0).numpy(), expected)


def test_resnet18_four_block_subgraph_and_profiler_metrics():
    """Run one ResNet-18-sized four-block stage with measurable VTA work."""
    env = vta.get_env()
    shape = (1, env.BLOCK_IN, 4, 4)
    data_var = relay.var("data", shape=shape, dtype="int8")
    identity = np.zeros((env.BLOCK_OUT, env.BLOCK_IN, 1, 1), dtype="int8")
    for channel in range(env.BLOCK_IN):
        identity[channel, channel, 0, 0] = 1
    value = data_var
    for _ in range(4):
        branch = _conv2d(_conv2d(value, identity), identity)
        value = relay.annotation.stop_fusion(relay.add(branch, value))
    mod = tvm.IRModule.from_expr(relay.Function([data_var], value))
    partitioned = vta.partition_for_vta(mod)
    regions = [
        function for function in partitioned.functions.values()
        if isinstance(function, relay.Function) and function.attrs
        and function.attrs.get("Compiler") == "vta"
    ]
    assert len(regions) == 4
    runtime = _build_runtime(mod)
    data = np.random.randint(-4, 5, shape).astype("int8")
    simulator.clear_stats()
    runtime.set_input("data", data)
    runtime.run()
    expected = (data.astype("int16") * 16).astype("int8")
    np.testing.assert_equal(runtime.get_output(0).numpy(), expected)
    stats = simulator.stats()
    assert stats["gemm_counter"] > 0
    assert stats["alu_counter"] > 0
    assert stats["inp_load_nbytes"] > 0
    assert stats["out_store_nbytes"] > 0
    print("ResNet-18 four-block FSIM metrics:", stats)


def test_acc8_runtime_uses_input_element_address_units():
    runtime_source = (
        __import__("pathlib").Path(vta.__file__).resolve().parents[2] / "runtime" / "runtime.cc"
    ).read_text(encoding="utf-8")
    assert "case VTA_MEM_ID_ACC_8BIT:\n        elem_bytes = VTA_INP_ELEM_BYTES;" in runtime_source


def test_fused_residual_removes_host_bridge_and_reduces_transfers():
    env = vta.get_env()
    shape = (1, env.BLOCK_IN, 4, 4)
    data_var = relay.var("data", shape=shape, dtype="int8")
    identity = np.zeros((env.BLOCK_OUT, env.BLOCK_IN, 1, 1), dtype="int8")
    for channel in range(env.BLOCK_IN):
        identity[channel, channel, 0, 0] = 1

    fused = relay.add(_conv2d(_conv2d(data_var, identity), identity), data_var)
    fused_mod = tvm.IRModule.from_expr(relay.Function([data_var], fused))
    fused_runtime = _build_runtime(fused_mod)
    data = np.random.randint(-16, 17, shape).astype("int8")
    simulator.clear_stats()
    fused_runtime.set_input("data", data)
    fused_runtime.run()
    fused_stats = simulator.stats()

    split_first = relay.annotation.stop_fusion(_conv2d(data_var, identity))
    split_second = relay.annotation.stop_fusion(_conv2d(split_first, identity))
    split_mod = tvm.IRModule.from_expr(relay.Function([data_var], relay.add(split_second, data_var)))
    split_runtime = _build_runtime(split_mod)
    simulator.clear_stats()
    split_runtime.set_input("data", data)
    split_runtime.run()
    split_stats = simulator.stats()

    np.testing.assert_equal(fused_runtime.get_output(0).numpy(), split_runtime.get_output(0).numpy())
    assert fused_stats["acc_load_nbytes"] < split_stats["acc_load_nbytes"]
    codegen_source = (__import__("pathlib").Path(vta.__file__).resolve().parent /
                      "compiler" / "codegen.py").read_text(encoding="utf-8")
    residual_source = codegen_source.split("def _residual_conv_wrapper", 1)[1].split(
        "def _add_wrapper", 1
    )[0]
    assert "habranch" not in residual_source
    assert "host_bridge_count=0" in residual_source


def test_complete_resnet18_mixed_graph_executor_fsim():
    """Eight basic blocks; projection boundaries use correct multi-region fallback."""
    env = vta.get_env()
    data_var = relay.var("data", shape=(1, env.BLOCK_IN, 4, 4), dtype="int8")
    value = data_var
    channels = env.BLOCK_IN
    for stage in range(4):
        target_channels = env.BLOCK_OUT * (2 ** stage)
        if stage:
            main1 = np.zeros((target_channels, channels, 3, 3), dtype="int8")
            main2 = np.zeros((target_channels, target_channels, 3, 3), dtype="int8")
            projection = np.zeros((target_channels, channels, 1, 1), dtype="int8")
            for channel in range(channels):
                main1[channel, channel, 1, 1] = 1
                projection[channel, channel, 0, 0] = 1
            for channel in range(target_channels):
                main2[channel, channel, 1, 1] = 1
            branch = relay.annotation.stop_fusion(
                _conv2d(_conv2d(value, main1, stride=2, padding=1), main2, padding=1)
            )
            shortcut = relay.annotation.stop_fusion(_conv2d(value, projection, stride=2))
            value = relay.annotation.stop_fusion(relay.add(branch, shortcut))
            channels = target_channels
        identity = np.zeros((channels, channels, 1, 1), dtype="int8")
        for channel in range(channels):
            identity[channel, channel, 0, 0] = 1
        for _ in range(2 if stage == 0 else 1):
            value = relay.annotation.stop_fusion(
                relay.add(_conv2d(_conv2d(value, identity), identity), value)
            )
    pooled = relay.mean(value.astype("int32"), axis=[2, 3])
    flattened = relay.reshape(pooled, (1, channels))
    logits = relay.nn.dense(flattened.astype("float32"), relay.const(
        np.ones((4, channels), dtype="float32")
    ))
    mod = tvm.IRModule.from_expr(relay.Function([data_var], logits))
    runtime = _build_runtime(mod)
    data = np.random.randint(-2, 3, (1, env.BLOCK_IN, 4, 4)).astype("int8")
    runtime.set_input("data", data)
    runtime.run()
    output = runtime.get_output(0).numpy()
    assert output.shape == (1, 4)
    assert np.isfinite(output).all()
