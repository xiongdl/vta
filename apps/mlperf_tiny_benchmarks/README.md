# MLPerf Tiny benchmark models

`pretrainedResnet_quant.tflite` is the quantized ResNet8 image-classification
model from MLCommons MLPerf Tiny commit
`bceb91c5ad2e2deb295547d81505721d3a87d578`.

- Upstream file: `benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite`
- SHA-256: `3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415`
- Upstream project license: Apache-2.0

The VTA test extracts operator 10, a 1x1 stride-2 projection QConv, and compares
the standalone VTA result with the corresponding LiteRT intermediate tensor. It
does not execute the residual block.

Run from the `tvm-vta` directory after initializing the project environment:

```bash
pytest -q tests/python/test_tflite_resnet8_conv.py
```

Set `VTA_RESNET8_TFLITE` only when intentionally testing another compatible
model file.
