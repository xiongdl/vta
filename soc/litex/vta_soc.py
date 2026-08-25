#!/usr/bin/env python3
"""Generate the first LiteX CPU + frozen-VTA SoC for ZCU104."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from litex.soc.integration.builder import Builder
from litex.soc.integration.soc import SoCRegion
from litex.build.sim.config import SimConfig
from litex.tools.litex_sim import SimSoC
from litex_boards.targets.xilinx_zcu104 import BaseSoC

from vta_ip import FrozenVTA


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("soc_config", ROOT / "tools" / "soc_config.py")
SOC_CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOC_CONFIG)


class ZCU104VTASoC(BaseSoC):
    def __init__(self, config):
        if config["vta"]["vcr_frontend"] != "apb32":
            raise ValueError("LiteX system integration currently supports vcr_frontend=apb32; AHB32 bridge RTL is staged separately")
        super().__init__(
            sys_clk_freq=config["soc"]["clock_hz"],
            cpu_type=config["soc"]["cpu"],
            bus_standard="axi",
            bus_data_width=config["vta"]["memory_data_width"],
            bus_address_width=config["soc"]["address_width"],
            integrated_rom_size=SOC_CONFIG.number(config["rom"]["size"]),
            integrated_sram_size=SOC_CONFIG.number(config["sram"]["size"]),
            integrated_main_ram_size=0,
            with_uart=True,
            with_timer=True,
            with_led_chaser=True,
        )
        self.submodules.vta = vta = FrozenVTA(self.platform, config["_vta_ip_dir"])
        self.bus.add_slave("vta_vcr", vta.control.wb, SoCRegion(
            origin=SOC_CONFIG.number(config["vta"]["vcr_base"]),
            size=SOC_CONFIG.number(config["vta"]["vcr_size"]),
            cached=False))
        self.bus.add_master("vta_dma", vta.axi)
        self.add_constant("VTA_VCR_BASE", SOC_CONFIG.number(config["vta"]["vcr_base"]))
        self.add_constant("VTA_MEMORY_DATA_WIDTH", config["vta"]["memory_data_width"])


class SimVTASoC(SimSoC):
    def __init__(self, config):
        if config["vta"]["vcr_frontend"] != "apb32":
            raise ValueError("the first simulation target supports vcr_frontend=apb32")
        super().__init__(
            with_sdram=True,
            sdram_data_width=config["ddr"]["data_width"],
            cpu_type=config["soc"]["cpu"],
            bus_standard="axi",
            # Keep the RV32 CPU/CSR path at 32 bits.  A 64-bit AXI system bus
            # turns 32-bit CSR accesses at address +4 into misaligned 64-bit
            # transactions with the current LiteX AXI-to-CSR bridge.
            # LiteX inserts a width converter for the native VTA AXI master.
            bus_data_width=32,
            bus_address_width=config["soc"]["address_width"],
            integrated_rom_size=SOC_CONFIG.number(config["rom"]["size"]),
            integrated_sram_size=SOC_CONFIG.number(config["sram"]["size"]),
            integrated_main_ram_size=0,
            uart_name="sim",
            with_timer=True,
        )
        self.submodules.vta = vta = FrozenVTA(self.platform, config["_vta_ip_dir"])
        self.bus.add_slave("vta_vcr", vta.control.wb, SoCRegion(
            origin=SOC_CONFIG.number(config["vta"]["vcr_base"]),
            size=SOC_CONFIG.number(config["vta"]["vcr_size"]),
            cached=False))
        self.bus.add_master("vta_dma", vta.axi)
        self.add_constant("VTA_VCR_BASE", SOC_CONFIG.number(config["vta"]["vcr_base"]))
        self.add_constant("VTA_MEMORY_DATA_WIDTH", config["vta"]["memory_data_width"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("zcu104",), default="zcu104")
    parser.add_argument("--build", action="store_true", help="run Vivado after generation")
    parser.add_argument("--sim", action="store_true", help="reserved for the dedicated Verilator target")
    args = parser.parse_args()
    config = SOC_CONFIG.validate(SOC_CONFIG.load_config(args.config), require_ip=True)
    soc = SimVTASoC(config) if args.sim else ZCU104VTASoC(config)
    builder = Builder(
        soc,
        output_dir=str(args.build_dir),
        csr_csv=str(args.build_dir / "csr.csv"),
        compile_software=args.build,
        compile_gateware=args.build,
    )
    if args.sim:
        sim_config = SimConfig()
        sim_config.add_clocker("sys_clk", freq_hz=int(1e6))
        sim_config.add_module("serial2console", "serial")
        builder.build(sim_config=sim_config, run=args.build)
    else:
        builder.build(run=args.build)


if __name__ == "__main__":
    main()
