# Out-of-tree VTA compiler and runtime

The new integration keeps VTA-specific compilation, configuration, and
simulation in this repository. Apache TVM is built with `USE_VTA_FSIM`,
`USE_VTA_TSIM`, and `USE_VTA_FPGA` disabled.

## Python compiler passes

Use `tvm-vta/python` ahead of TVM's legacy `vta/python` directory:

```bash
export PYTHONPATH="$PWD/python:/path/to/tvm/python"
```

The first graphpack-free entry point is:

```python
import vta

partitioned = vta.partition_for_vta(mod, params=params)
```

The first executable Relay path supports a single symmetric int8 QNN dense
region:

```python
artifact = vta.compile_partitioned_dense(partitioned)
```

It performs compile-time weight packing and lowers through VTA GEMM
tensorization and the standalone runtime. Non-zero zero-points, non-unit
scales, dynamic shapes, and dimensions not aligned to the hardware blocks are
currently rejected explicitly.

It uses `MergeComposite`, `AnnotateTarget`, `MergeCompilerRegions`, and
`PartitionGraph`. Packing and VTA Relay-to-TIR lowering are separate compiler
stages and are not implemented by selecting operator names in the source graph.

## Standalone functional simulator

```bash
cmake -S . -B build-fsim \
  -DTVM_SOURCE_DIR=/path/to/tvm \
  -DTVM_BUILD_DIR=/path/to/tvm/build \
  -DVTA_RUNTIME=sim
cmake --build build-fsim
```

On macOS, use `CC=/usr/bin/clang CXX=/usr/bin/clang++` when the Conda LLVM
toolchain is older than the installed Command Line Tools SDK. Set
`VTA_LIBRARY_PATH` to the standalone build directory when loading the plugin.
`VTA_CONFIG` can select an absolute configuration file without editing the
tracked default configuration.

TSIM uses the same standalone boundary but additionally requires the generated
RTL model and Verilator integration; that build target will be completed after
the independent FSIM target and Relay-to-TIR compiler baseline are stable.
