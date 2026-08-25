# LiteX Mini VTA SoC

This directory builds a minimal CPU + VTA system for Verilator and ZCU104.
The VTA Chisel sources are outside the SoC build boundary: a SoC build consumes
only a frozen generated-RTL package with a checked manifest and SHA-256 digest.

## Baseline

- VexRiscv RV32IM CPU, no cache.
- 64 KiB boot ROM and configurable on-chip SRAM (512 KiB by default).
- UART, timer and interrupt support supplied by LiteX.
- Native VTA APB4 32-bit control port.
- Optional AHB-Lite 32-bit VCR access through `rtl/amba/ahb32_to_apb32.sv`.
- Selectable native VTA memory port: AHB-Lite 32/64-bit or AXI4 32/64-bit.
- AXI4 64-bit is the first complete SIM/ZCU104 path.

## Build boundary

Generating VTA RTL is an explicit release-preparation operation and is not part
of any target in this Makefile. Import an already generated top-level file:

```bash
python3 tools/import_vta_ip.py \
  --variant apb32-axi64 \
  --rtl /path/to/generated/VTAShellAPB.v \
  --vta-config ../config/vta_config.json \
  --output-root ../generated/vta-ip
```

The importer rejects a mismatched top module or memory width. Normal builds
validate the imported manifest and never access `../hardware/chisel`.

## Configuration and local checks

```bash
make test
python3 tools/soc_config.py check \
  --config config/sim_default.json --allow-missing-ip
python3 tools/soc_config.py generate \
  --config config/sim_default.json --allow-missing-ip \
  --output build/generated
```

`make sim` and `make zcu104` intentionally require an imported VTA IP. ZCU104
gateware also requires a Linux host with AMD Vivado; the macOS development host
can run configuration tests and Verilator once LiteX dependencies are present.

For an offline installation, cache every package in `requirements.txt`, including
`meson` and `ninja`. LiteX uses them to configure and build Picolibc before the
RISC-V BIOS is linked; generating the RTL alone does not exercise this dependency.
The local `.venv` is created with the Python 3.10 interpreter from the
`tvm_py310` Conda environment. On macOS, LiteX simulation also needs the native
`libevent` and `json-c` libraries; this workspace obtains them from the same
Conda environment as Verilator.

## Implemented baseline

The current baseline imports a frozen `VTAShellAPB` APB32 + AXI64 package and
elaborates a ZCU104 design containing VexRiscv, 64 KiB ROM, 512 KiB on-chip
SRAM, UART, timer, ZCU104 DDR and VTA. The simulation target keeps the RV32/CSR
AXI path at 32 bits: with the pinned LiteX version, a 64-bit AXI system bus
turns RV32 accesses to CSR offsets such as `+4` into unsupported/misaligned
64-bit transactions. VTA remains a native AXI64 master and LiteX inserts a
width converter on the simulation target. The ZCU104 target retains a native
64-bit system/memory interconnect for performance validation. Generate the
Vivado project without running Vivado with:

```bash
.venv/bin/python litex/vta_soc.py \
  --config config/zcu104_axi64.json \
  --build-dir build/zcu104-axi64
```

The APB32 firmware driver and standalone AHB32-to-APB32 bridge are present.
The 32-bit simulation system has executed the LiteX BIOS banner under
Verilator. On macOS, `make sim-run` applies the pinned LiteX compatibility patch,
builds with the local RISC-V toolchain and system clang++, and starts the BIOS.
After a model has been built, `make vcr-smoke` writes and reads VTA register
offset `0x08` through the CPU and checks the exact value. AHB VCR insertion into
LiteX, VTA workload execution, AHB/AXI32 memory
variants, NN package loading and the actual ZCU104 bitstream remain required
before the system can be described as inference-ready.

## FSIM/TSIM case export and SoC replay

The simulator drivers have an opt-in raw DRAM export hook. Set
`VTA_CASE_DUMP_DIR` before a workload, or call `VTAExportCase()` immediately
before and after `VTADeviceRun()`. FSIM and TSIM then emit the same
`before_manifest.json`/`after_manifest.json` plus allocation binaries. Normal
execution is unchanged when the variable is absent.

Rebuild the runtime for the backend selected by `config/vta_config.json` before
capturing an in-tree Python integration test. In particular, the repository
default selects TSIM, so rebuilding only `vta_fsim` will leave the integration
process using an older `libvta_tsim` without the export hook:

```bash
cmake --build ../../tvm/build --target vta_tsim -j4
VTA_CASE_DUMP_DIR=/tmp/vta-raw python ../../tvm/vta/tests/python/integration/test_benchmark_topi_dense.py
```

Convert a raw dump into a relocatable SoC package with:

```bash
.venv/bin/python tools/vta_case.py pack /path/to/raw /path/to/case
.venv/bin/python tools/vta_case.py validate /path/to/case
```

The packer uses the active `vta_config.json` and a helper compiled against
`include/vta/hw_spec.h`; it does not duplicate or change the instruction ABI.
It rewrites only `VTAMemInsn.dram_base`, changing simulator absolute element
addresses into offsets relative to the VCR base for each memory type. ACC and
ACC-8 loads share one aligned `acc.bin`, so bias and per-channel ALU parameter
allocations (for example multiplier and shift planes) are delivered as one
array without requiring an RTL or `hw_spec.h` field change. `manifest.json`
records every access, section hash and expected post-run output.

A minimal direct-FSIM capture, useful for checking the export path independently
of TVM's dynamic-library search order, is:

```bash
python tools/fsim_case_smoke.py /tmp/vta-raw \
  --library ../build-standalone-fsim-case/libvta_fsim_case.dylib
python tools/vta_case.py pack /tmp/vta-raw /tmp/vta-case
```

Replay a package through an existing LiteX model with:

```bash
make case-run CASE=/tmp/vta-case BUILD_DIR=build
```

The runner loads all manifest sections, flushes CPU/L2 caches, programs the
seven VCR workload registers, starts VTA and checks every expected output span.
