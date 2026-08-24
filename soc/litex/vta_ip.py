"""LiteX wrapper for a frozen APB32 + AXI VTA RTL package."""

import json
from pathlib import Path

from migen import ClockSignal, Constant, FSM, If, Instance, Module, NextState, ResetSignal, Signal
from litex.soc.interconnect import axi, wishbone


class WishboneToAPB32(Module):
    """Single-request Wishbone to APB4 bridge used for the VTA VCR."""

    def __init__(self, address_width=16):
        self.wb = wishbone.Interface(data_width=32, adr_width=30, addressing="word")
        self.paddr = Signal(address_width)
        self.psel = Signal()
        self.penable = Signal()
        self.pwrite = Signal()
        self.pwdata = Signal(32)
        self.pstrb = Signal(4)
        self.pprot = Signal(3)
        self.prdata = Signal(32)
        self.pready = Signal()
        self.pslverr = Signal()

        self.submodules.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(self.wb.cyc & self.wb.stb,
                self.paddr.eq(self.wb.adr[: max(0, address_width - 2)] << 2),
                self.pwrite.eq(self.wb.we),
                self.pwdata.eq(self.wb.dat_w),
                self.pstrb.eq(self.wb.sel),
                self.pprot.eq(0),
                NextState("SETUP")))
        fsm.act("SETUP", self.psel.eq(1), NextState("ACCESS"))
        fsm.act("ACCESS",
            self.psel.eq(1), self.penable.eq(1),
            If(self.pready,
                self.wb.dat_r.eq(self.prdata),
                self.wb.ack.eq(~self.pslverr),
                self.wb.err.eq(self.pslverr),
                NextState("IDLE")))


class FrozenVTA(Module):
    def __init__(self, platform, ip_dir):
        ip_dir = Path(ip_dir)
        manifest = json.loads((ip_dir / "manifest.json").read_text())
        if manifest["host_protocol"] != "apb4" or manifest["host_data_width"] != 32:
            raise ValueError("FrozenVTA requires an APB4 32-bit host package")
        if manifest["memory_protocol"] != "axi4":
            raise ValueError("the first LiteX integration supports native AXI4 VTA memory")
        width = manifest["memory_data_width"]
        if width not in (32, 64):
            raise ValueError("VTA AXI data width must be 32 or 64")

        for entry in (ip_dir / "filelist.f").read_text().splitlines():
            entry = entry.strip()
            if entry and not entry.startswith("#"):
                platform.add_source(str(ip_dir / entry))

        self.submodules.control = control = WishboneToAPB32(address_width=16)
        self.axi = axi.AXIInterface(data_width=width, address_width=32, id_width=8)
        a = self.axi
        aw_len = Signal(4)
        ar_len = Signal(4)
        aw_lock = Signal(2)
        ar_lock = Signal(2)
        self.comb += [
            a.aw.len.eq(aw_len),
            a.ar.len.eq(ar_len),
            a.aw.lock.eq(aw_lock[0]),
            a.ar.lock.eq(ar_lock[0]),
        ]

        params = dict(
            i_clock=ClockSignal("sys"),
            i_reset=ResetSignal("sys"),
            i_io_host_paddr=control.paddr,
            i_io_host_psel=control.psel,
            i_io_host_penable=control.penable,
            i_io_host_pwrite=control.pwrite,
            i_io_host_pwdata=control.pwdata,
            i_io_host_pstrb=control.pstrb,
            i_io_host_pprot=control.pprot,
            o_io_host_prdata=control.prdata,
            o_io_host_pready=control.pready,
            o_io_host_pslverr=control.pslverr,
            i_io_mem_aw_ready=a.aw.ready,
            o_io_mem_aw_valid=a.aw.valid,
            o_io_mem_aw_bits_addr=a.aw.addr,
            o_io_mem_aw_bits_id=a.aw.id,
            o_io_mem_aw_bits_len=aw_len,
            o_io_mem_aw_bits_size=a.aw.size,
            o_io_mem_aw_bits_burst=a.aw.burst,
            o_io_mem_aw_bits_lock=aw_lock,
            o_io_mem_aw_bits_cache=a.aw.cache,
            o_io_mem_aw_bits_prot=a.aw.prot,
            o_io_mem_aw_bits_qos=a.aw.qos,
            i_io_mem_w_ready=a.w.ready,
            o_io_mem_w_valid=a.w.valid,
            o_io_mem_w_bits_data=a.w.data,
            o_io_mem_w_bits_strb=a.w.strb,
            o_io_mem_w_bits_last=a.w.last,
            o_io_mem_w_bits_id=Signal(8),
            o_io_mem_b_ready=a.b.ready,
            i_io_mem_b_valid=a.b.valid,
            i_io_mem_b_bits_resp=a.b.resp,
            i_io_mem_b_bits_id=a.b.id,
            i_io_mem_ar_ready=a.ar.ready,
            o_io_mem_ar_valid=a.ar.valid,
            o_io_mem_ar_bits_addr=a.ar.addr,
            o_io_mem_ar_bits_id=a.ar.id,
            o_io_mem_ar_bits_len=ar_len,
            o_io_mem_ar_bits_size=a.ar.size,
            o_io_mem_ar_bits_burst=a.ar.burst,
            o_io_mem_ar_bits_lock=ar_lock,
            o_io_mem_ar_bits_cache=a.ar.cache,
            o_io_mem_ar_bits_prot=a.ar.prot,
            o_io_mem_ar_bits_qos=a.ar.qos,
            o_io_mem_r_ready=a.r.ready,
            i_io_mem_r_valid=a.r.valid,
            i_io_mem_r_bits_data=a.r.data,
            i_io_mem_r_bits_resp=a.r.resp,
            i_io_mem_r_bits_last=a.r.last,
            i_io_mem_r_bits_id=a.r.id,
        )
        # Chisel's coherent/user fields are constants for VTA and are not
        # represented by LiteX's generic AXI interface.
        for name in ("aw", "w", "ar"):
            params[f"o_io_mem_{name}_bits_user"] = Signal(5)
        params["o_io_mem_aw_bits_region"] = Signal(4)
        params["i_io_mem_b_bits_user"] = Constant(0, 5)
        params["o_io_mem_ar_bits_region"] = Signal(4)
        params["i_io_mem_r_bits_user"] = Constant(0, 5)
        self.specials += Instance(manifest["top"], **params)
