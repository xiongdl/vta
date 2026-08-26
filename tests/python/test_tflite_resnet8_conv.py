"""Bit-exact VTA execution of the first MLPerf Tiny ResNet8 TFLite QConv."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tvm
from tvm import relay, rpc
from tvm.contrib import utils
from tvm.topi.testing import conv2d_nchw_python

import vta
from vta.testing import simulator  # noqa: F401


def _tensor_data(model, subgraph, index, dtype):
    tensor = subgraph.Tensors(index)
    shape = tuple(tensor.Shape(i) for i in range(tensor.ShapeLength()))
    raw = model.Buffers(tensor.Buffer()).DataAsNumpy().tobytes()
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _quant(tensor):
    quant = tensor.Quantization()
    scale = np.asarray([quant.Scale(i) for i in range(quant.ScaleLength())], dtype="float64")
    zero = np.asarray([quant.ZeroPoint(i) for i in range(quant.ZeroPointLength())], dtype="int32")
    return scale, zero


def _pack_input(data, artifact, env):
    n, channels, height, width = data.shape
    padded_channels = artifact.packed_input_shape[1] * env.BLOCK_IN
    padded = np.zeros((n, padded_channels, height, width), dtype="int8")
    padded[:, :channels] = data
    return padded.reshape(n // env.BATCH, env.BATCH, padded_channels // env.BLOCK_IN,
                          env.BLOCK_IN, height, width).transpose(0, 2, 4, 5, 1, 3)


def _unpack_output(data, artifact, env):
    n, channels, height, width = artifact.output_shape
    physical = artifact.packed_output_shape[1] * env.BLOCK_OUT
    return data.transpose(0, 4, 1, 5, 2, 3).reshape(
        n, physical, height, width)[:, :channels]


def test_mlperf_tiny_resnet8_qconv_tflite_exact():
    tflite = pytest.importorskip("tflite")
    litert = pytest.importorskip("ai_edge_litert.interpreter")
    default_model = (Path(__file__).resolve().parents[2] / "apps" /
                     "mlperf_tiny_benchmarks" / "pretrainedResnet_quant.tflite")
    model_path = Path(os.environ.get("VTA_RESNET8_TFLITE", default_model))
    if not model_path.is_file():
        pytest.skip(f"MLPerf Tiny ResNet8 model not found: {model_path}")
    model = tflite.Model.GetRootAsModel(Path(model_path).read_bytes(), 0)
    subgraph = model.Subgraphs(0)
    # Operator 10 is the real 1x1, stride-2 projection Conv in the last stage.
    # Run it alone: no residual ADD and no quantized spatial-padding ambiguity.
    operator = subgraph.Operators(10)
    input_index, weight_index, bias_index = (operator.Inputs(i) for i in range(3))
    output_index = operator.Outputs(0)

    weight_ohwi = _tensor_data(model, subgraph, weight_index, "int8")
    weight = weight_ohwi.transpose(0, 3, 1, 2)
    bias = _tensor_data(model, subgraph, bias_index, "int32")
    input_scale, input_zero = _quant(subgraph.Tensors(input_index))
    weight_scale, weight_zero = _quant(subgraph.Tensors(weight_index))
    output_scale, output_zero = _quant(subgraph.Tensors(output_index))
    assert weight_zero.size == weight.shape[0] and np.all(weight_zero == 0)

    input_nhwc = tuple(subgraph.Tensors(input_index).Shape(i)
                       for i in range(subgraph.Tensors(input_index).ShapeLength()))
    output_nhwc = tuple(subgraph.Tensors(output_index).Shape(i)
                        for i in range(subgraph.Tensors(output_index).ShapeLength()))
    n, height, width, channels = input_nhwc
    out_channels = output_nhwc[3]
    kernel_h, kernel_w = weight.shape[2:]
    data_var = relay.var("data", shape=(n, channels, height, width), dtype="int8")
    conv = relay.qnn.op.conv2d(
        data_var, relay.const(weight), relay.const(int(input_zero[0]), "int32"),
        relay.const(0, "int32"), relay.const(float(input_scale[0]), "float32"),
        relay.const(weight_scale.astype("float32")), strides=(2, 2), padding=(0, 0),
        channels=out_channels, kernel_size=(kernel_h, kernel_w), data_layout="NCHW",
        kernel_layout="OIHW", out_dtype="int32")
    biased = relay.nn.bias_add(conv, relay.const(bias), axis=1)
    # TVM 0.17 requires scalar qnn.requantize scales. The compiler receives the
    # original per-channel TFLite ratios explicitly below.
    requantized = relay.qnn.op.requantize(
        biased, relay.const(float(input_scale[0] * weight_scale[0]), "float32"),
        relay.const(0, "int32"), relay.const(float(output_scale[0]), "float32"),
        relay.const(int(output_zero[0]), "int32"), axis=1, out_dtype="int8")
    function = relay.Function([data_var], relay.clip(requantized, a_min=-128, a_max=127))
    ratios = input_scale[0] * weight_scale / output_scale[0]
    artifact = vta.compile_qnn_conv2d(
        function, name="mlperf_resnet8_conv10", requant_ratios=ratios)

    # Run the complete TFLite graph only to obtain the real input/output tensors
    # of operator 10. VTA still executes operator 10 alone.
    interpreter = litert.Interpreter(
        model_path=str(model_path), experimental_preserve_all_tensors=True,
        experimental_op_resolver_type=litert.OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    model_input = np.random.default_rng(0).integers(
        -128, 128, size=tuple(interpreter.get_input_details()[0]["shape"]), dtype="int8")
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], model_input)
    interpreter.invoke()
    logical_input = interpreter.get_tensor(input_index).transpose(0, 3, 1, 2).copy()
    tflite_expected = interpreter.get_tensor(output_index).transpose(0, 3, 1, 2).copy()
    env = vta.get_env()
    packed_input = _pack_input(logical_input, artifact, env)
    temp = utils.tempdir()
    artifact.module.save(temp.relpath("conv.o"))
    remote = rpc.LocalSession()
    remote.upload(temp.relpath("conv.o"))
    module = remote.load_module("conv.o")
    device = remote.ext_dev(0)
    output = tvm.nd.empty(artifact.packed_output_shape, env.out_dtype, device)
    arguments = [tvm.nd.array(packed_input, device),
                 tvm.nd.array(artifact.packed_weight, device),
                 tvm.nd.array(artifact.packed_bias, device)]
    arguments += [tvm.nd.array(value, device) for value in artifact.packed_requant]
    module(*arguments, output)
    actual = _unpack_output(output.numpy(), artifact, env)

    centered = logical_input.astype("int32") - int(input_zero[0])
    accumulated = conv2d_nchw_python(centered, weight.astype("int32"), 2, 0)
    accumulated += bias.reshape(1, -1, 1, 1)
    expected = np.empty_like(accumulated, dtype="int32")
    for channel, ratio in enumerate(ratios):
        multiplier, shift = vta.quantize_multiplier(float(ratio))
        expected[:, channel] = np.vectorize(
            lambda value: vta.cmsis_nn_requantize(value, multiplier, shift),
            otypes=[np.int32])(accumulated[:, channel])
    expected = np.clip(expected + int(output_zero[0]), -128, 127).astype("int8")
    np.testing.assert_array_equal(expected, tflite_expected)
    np.testing.assert_array_equal(actual, expected)
