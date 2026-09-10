# MLPerf Tiny ResNet-8 source model

`pretrainedResnet.tflite` is an unmodified floating-point model copied
byte-for-byte from the MLPerf Tiny v1.4 source tree at:

```text
benchmark/training/image_classification/trained_models/pretrainedResnet.tflite
```

Its SHA-256 is:

```text
b5c0046d6e0328b4956afd6baa29555a29b1f1c65bdd45aaed75b7cd484d9f79
```

The FlatBuffer has one `float32` NHWC input of shape `[1, 32, 32, 3]`, one
`float32` output of shape `[1, 10]`, and nine convolutions whose output-channel
progression is `16/16/16/32/32/32/64/64/64`. The model is distributed by
MLPerf Tiny under Apache-2.0; the applicable text is copied verbatim to
`../LICENSE.mlperf-tiny`.

The committed bytes remain floating point. Quantization is performed by TVM at
deployment time with the following fixed policy:

```python
with relay.quantize.qconfig(
    calibrate_mode="global_scale",
    global_scale=8.0,
    skip_conv_layers=[0],
):
    quantized = relay.quantize.quantize(mod, params=params)
```

CIFAR-10 samples are not calibration input.
