#!/usr/bin/env python3
"""Emit a simple bare-metal C/H VTA SoC case from a packed case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ("insn", "uop", "inp", "wgt", "acc", "out")
INST_BITS = 128
MEMORY_BURST_BEATS = 16


def alignment_for(name: str, manifest: dict) -> int:
    data_bits = 1 << manifest["config"]["LOG_BUS_WIDTH"]
    if data_bits < 32 or data_bits & (data_bits - 1):
        raise ValueError(f"unsupported VTA memory data width: {data_bits}")
    return MEMORY_BURST_BEATS * (data_bits // 8) if name == "insn" else 8


def section_for(name: str) -> str:
    if name in ("insn", "uop", "wgt", "acc"):
        return ".rodata_ai"
    return ".data_ai.out" if name == "out" else ".data_ai.init"


def words(data: bytes, width: int) -> str:
    data += bytes((-len(data)) % width)
    suffix = "ULL" if width == 8 else "U"
    digits = width * 2
    values = [f"0x{int.from_bytes(data[i:i + width], 'little'):0{digits}x}{suffix}"
              for i in range(0, len(data), width)]
    return "\n".join("    " + ", ".join(values[i:i + 4]) + ","
                     for i in range(0, len(values), 4))


def emit_header(case: Path, output: Path, symbol: str) -> None:
    manifest = json.loads((case / "manifest.json").read_text())
    if manifest.get("baddr_mode", "add") != "add":
        raise ValueError("C firmware generation requires additive VTA baddr RTL")
    definitions = []
    for name in ORDER:
        section = manifest["sections"].get(name)
        if section is None:
            continue
        data = (case / section["file"]).read_bytes()
        bus_bytes = (1 << manifest["config"]["LOG_BUS_WIDTH"]) // 8
        ctype, width = (("uint32_t", 4) if name == "uop" or bus_bytes == 4
                        else ("uint64_t", 8))
        qualifier = "const " if section_for(name) == ".rodata_ai" else ""
        definitions.append(
            f"{qualifier}{ctype} {symbol}_{name}[]\n"
            f"    VTA_CASE_SECTION(\"{section_for(name)}\", {alignment_for(name, manifest)}) = {{\n"
            f"{words(data, width)}\n}};\n")
    expected = (case / manifest["sections"]["expected_out"]["file"]).read_bytes()
    definitions.append(
        f"uint64_t {symbol}_expected[]\n"
        f"    VTA_CASE_SECTION(\".data_ai.expected\", 8) = {{\n"
        f"{words(expected, 8)}\n}};\n")
    config = manifest["config"]
    output.write_text(f"""#include <stddef.h>
#include <stdint.h>

#if defined(__APPLE__)
#define VTA_CASE_SECTION(name, alignment) __attribute__((aligned(alignment)))
#else
#define VTA_CASE_SECTION(name, alignment) \\
    __attribute__((section(name), aligned(alignment)))
#endif

/* Generated VTA single-operator SoC case.
 * TARGET={config['TARGET']}, HW_VER={config['HW_VER']}, LOG_BUS_WIDTH={config['LOG_BUS_WIDTH']}
 * Each 128-bit instruction is stored as two little-endian uint64_t words.
 * Include this data header from this case's C file only.
 */

{chr(10).join(definitions)}
#define {symbol.upper()}_INSN_COUNT \\
    ((uint32_t)(sizeof({symbol}_insn) / 16U))
#define {symbol.upper()}_EXPECTED_SIZE \\
    ((size_t)sizeof({symbol}_expected))

_Static_assert(sizeof({symbol}_insn) % 16U == 0U,
               "VTA instruction array is not a multiple of 128 bits");
_Static_assert(sizeof({symbol}_out) >= sizeof({symbol}_expected),
               "VTA output buffer is smaller than expected output");
""")


def emit_test(output: Path, symbol: str, header: str) -> None:
    upper = symbol.upper()
    output.write_text(f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sim_debug.h>

#include "{header}"

#define NPU_BASE              0xf1000000U
#define VTA_CONTROL_REG       (NPU_BASE + 0x00U)
#define VTA_CYCLE_COUNT_REG   (NPU_BASE + 0x04U)
#define VTA_INSN_COUNT_REG    (NPU_BASE + 0x08U)
#define VTA_INSN_BASE_REG     (NPU_BASE + 0x0cU)
#define VTA_UOP_BASE_REG      (NPU_BASE + 0x10U)
#define VTA_INP_BASE_REG      (NPU_BASE + 0x14U)
#define VTA_WGT_BASE_REG      (NPU_BASE + 0x18U)
#define VTA_ACC_BASE_REG      (NPU_BASE + 0x1cU)
#define VTA_OUT_BASE_REG      (NPU_BASE + 0x20U)

#define MMIO32(address) (*(volatile uint32_t *)(uintptr_t)(address))
#define VTA_CONTROL_START 1U
#define VTA_STATUS_DONE   2U

int main(void) {{
    uint32_t insn = (uint32_t)(uintptr_t){symbol}_insn;
    uint32_t uop = (uint32_t)(uintptr_t){symbol}_uop;
    uint32_t inp = (uint32_t)(uintptr_t){symbol}_inp;
    uint32_t wgt = (uint32_t)(uintptr_t){symbol}_wgt;
    uint32_t acc = (uint32_t)(uintptr_t){symbol}_acc;
    uint32_t out = (uint32_t)(uintptr_t){symbol}_out;

    printf("VTA case {symbol}: insn=%u\\r\\n", (unsigned int){upper}_INSN_COUNT);
    printf("insn=%08x uop=%08x inp=%08x\\r\\n",
           (unsigned int)insn, (unsigned int)uop, (unsigned int)inp);
    printf("wgt=%08x acc=%08x out=%08x\\r\\n",
           (unsigned int)wgt, (unsigned int)acc, (unsigned int)out);

    MMIO32(VTA_INSN_COUNT_REG) = {upper}_INSN_COUNT;
    MMIO32(VTA_INSN_BASE_REG) = insn;
    MMIO32(VTA_UOP_BASE_REG) = uop;
    MMIO32(VTA_INP_BASE_REG) = inp;
    MMIO32(VTA_WGT_BASE_REG) = wgt;
    MMIO32(VTA_ACC_BASE_REG) = acc;
    MMIO32(VTA_OUT_BASE_REG) = out;
    MMIO32(VTA_CONTROL_REG) = VTA_CONTROL_START;

    while (MMIO32(VTA_CONTROL_REG) != VTA_STATUS_DONE) {{}}
    uint32_t cycles = MMIO32(VTA_CYCLE_COUNT_REG);
    printf("VTA case counters: cycles=%u insns=%u\\r\\n",
           (unsigned int)cycles, (unsigned int){upper}_INSN_COUNT);
    for (size_t offset = 0; offset < {upper}_EXPECTED_SIZE; offset += 4U) {{
        uint32_t actual = *(volatile uint32_t *)((uintptr_t)out + offset);
        uint32_t expected = *(const uint32_t *)((uintptr_t){symbol}_expected + offset);
        if (actual != expected) {{
            printf("VTA case FAIL: offset=0x%zx expected=%08x actual=%08x\\r\\n",
                   offset, (unsigned int)expected, (unsigned int)actual);
            sim_finish();
            return 1;
        }}
    }}
    printf("VTA case PASS\\r\\n");
    sim_finish();
    return 0;
}}
""")


def emit_makefile(output: Path, symbol: str) -> None:
    output.joinpath("Makefile").write_text(f'''# Generated single-operator VTA SoC case.
# LiteX invokes this file as the replacement BIOS package Makefile.
CASE_SOURCE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
SOC_DIR := $(abspath $(CASE_SOURCE_DIR)/../..)
ifeq ($(abspath $(CURDIR)),$(CASE_SOURCE_DIR))
.PHONY: rtl sim run
rtl:
\t$(MAKE) -C $(SOC_DIR) rtl CONFIG=config/sim_default.json
sim:
\t$(MAKE) -C $(SOC_DIR) sim CASE=$(CASE_SOURCE_DIR) CONFIG=config/sim_default.json
run:
\t$(MAKE) -C $(SOC_DIR) run CASE=$(CASE_SOURCE_DIR) CONFIG=config/sim_default.json
else
SOC_DIRECTORY := $(SOC_DIR)
BIOS_DIRECTORY := $(SOC_DIRECTORY)/.venv/lib/python3.10/site-packages/litex/soc/software/bios
include $(BIOS_DIRECTORY)/Makefile
LSCRIPT = ../../../../../../../../linker/vta_case.ld
CFLAGS += -I$(BIOS_DIRECTORY)
override VPATH := $(CASE_SOURCE_DIR):$(BIOS_DIRECTORY):$(BIOS_DIRECTORY)/cmds:$(CPU_DIRECTORY)
main.o: $(CASE_SOURCE_DIR)/{symbol}_test.c
	$(compile)
endif
''')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", default="vta")
    args = parser.parse_args()
    symbol = "".join(char if char.isalnum() else "_" for char in args.name)
    stem = f"{symbol}_test"
    args.output.mkdir(parents=True, exist_ok=True)
    emit_header(args.case.resolve(), args.output / f"{stem}.h", symbol)
    emit_test(args.output / f"{stem}.c", symbol, f"{stem}.h")
    emit_makefile(args.output, symbol)


if __name__ == "__main__":
    main()
