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

## Implemented baseline

The current baseline imports a frozen `VTAShellAPB` APB32 + AXI64 package and
elaborates a ZCU104 design containing VexRiscv, 64 KiB ROM, 512 KiB on-chip
SRAM, UART, timer, ZCU104 DDR and VTA. VTA is a native master on the 64-bit AXI
interconnect. Generate the Vivado project without running Vivado with:

```bash
.venv/bin/python litex/vta_soc.py \
  --config config/zcu104_axi64.json \
  --build-dir build/zcu104-axi64
```

The APB32 firmware driver and standalone AHB32-to-APB32 bridge are present.
Verilator full-SoC execution, AHB VCR insertion into LiteX, AHB/AXI32 memory
variants, NN package loading and the actual ZCU104 bitstream remain required
before the system can be described as inference-ready.
