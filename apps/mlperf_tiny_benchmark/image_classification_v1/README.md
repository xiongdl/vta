# MLPerf Tiny ResNet-8 HOST deployment

This fixed-purpose application imports the committed floating MLPerf Tiny v1.4
ResNet-8 model, applies the documented TVM quantization policy once, and builds
both a pure LLVM reference and a mixed VTA + LLVM Graph Executor artifact. It
reloads both host libraries, compares their output tensors exactly for the ten
committed PNG samples, and requires positive FSIM GEMM, weight-load, and
output-store activity.

Importing `vta` loads and validates the compiler target extension. The mixed
branch explicitly applies `vta.relay.partition_for_vta()` once, then passes
`tvm.target.Target("vta")` to `relay.build`; unsupported operators remain in the
LLVM host portion of the same standard runtime module.

The application is an execution-equivalence example. It does not report model
accuracy, performance, energy, or MLPerf submission results.

## Prerequisites

From the repository root, prepare the pinned Python environment and build the
compiler extension and FSIM libraries:

```bash
./scripts/setup_tvm_vta_env.sh
./scripts/build_vta_lib.sh --target libtvm-vta-ext
./scripts/build_vta_lib.sh --target libvta_fsim
```

## Run

From the repository root:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_config.json" \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v1/run.py
```

The optional `--output-dir PATH` changes only the generated-artifact location.
The default `build/` directory is ignored by the repository. Every run replaces
the deterministic `mlperf_resnet_llvm` and `mlperf_resnet_vta` host libraries
for the current platform. A partial export is removed if the build fails.

Successful output reports eight deterministic VTA regions, ten exact output
comparisons, and positive FSIM profiler counters. Missing libraries, unexpected
model or routing structure, output differences, and absent accelerator activity
cause a nonzero exit.
