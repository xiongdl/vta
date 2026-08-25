"""LiteX wrapper for a frozen APB32 + AXI VTA RTL package."""

import json
from pathlib import Path

from migen import ClockSignal, Constant, Display, FSM, If, Instance, Module, NextState, NextValue, ResetSignal, Signal
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
                # APB request fields must remain stable through SETUP and
                # ACCESS, after the originating Wishbone request is accepted.
                NextValue(self.paddr, self.wb.adr[: max(0, address_width - 2)] << 2),
                NextValue(self.pwrite, self.wb.we),
                NextValue(self.pwdata, self.wb.dat_w),
                NextValue(self.pstrb, self.wb.sel),
                NextValue(self.pprot, 0),
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
    def __init__(self, platform, ip_dir, sim_debug=False):
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
        # The current VME registers AR and holds VALID/payload until READY, so
        # the frozen IP can connect directly to the LiteX AXI interconnect.
        self.axi = raw = a = axi.AXIInterface(
            data_width=width, address_width=32, id_width=8)
        aw_len = Signal(4)
        ar_len = Signal(4)
        aw_lock = Signal(2)
        ar_lock = Signal(2)
        self.comb += [
            raw.aw.len.eq(aw_len),
            raw.ar.len.eq(ar_len),
            raw.aw.lock.eq(aw_lock[0]),
            raw.ar.lock.eq(ar_lock[0]),
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
            i_io_mem_aw_ready=raw.aw.ready,
            o_io_mem_aw_valid=raw.aw.valid,
            o_io_mem_aw_bits_addr=raw.aw.addr,
            o_io_mem_aw_bits_id=raw.aw.id,
            o_io_mem_aw_bits_len=aw_len,
            o_io_mem_aw_bits_size=raw.aw.size,
            o_io_mem_aw_bits_burst=raw.aw.burst,
            o_io_mem_aw_bits_lock=aw_lock,
            o_io_mem_aw_bits_cache=raw.aw.cache,
            o_io_mem_aw_bits_prot=raw.aw.prot,
            o_io_mem_aw_bits_qos=raw.aw.qos,
            i_io_mem_w_ready=raw.w.ready,
            o_io_mem_w_valid=raw.w.valid,
            o_io_mem_w_bits_data=raw.w.data,
            o_io_mem_w_bits_strb=raw.w.strb,
            o_io_mem_w_bits_last=raw.w.last,
            o_io_mem_w_bits_id=Signal(8),
            o_io_mem_b_ready=raw.b.ready,
            i_io_mem_b_valid=raw.b.valid,
            i_io_mem_b_bits_resp=raw.b.resp,
            i_io_mem_b_bits_id=raw.b.id,
            i_io_mem_ar_ready=raw.ar.ready,
            o_io_mem_ar_valid=raw.ar.valid,
            o_io_mem_ar_bits_addr=raw.ar.addr,
            o_io_mem_ar_bits_id=raw.ar.id,
            o_io_mem_ar_bits_len=ar_len,
            o_io_mem_ar_bits_size=raw.ar.size,
            o_io_mem_ar_bits_burst=raw.ar.burst,
            o_io_mem_ar_bits_lock=ar_lock,
            o_io_mem_ar_bits_cache=raw.ar.cache,
            o_io_mem_ar_bits_prot=raw.ar.prot,
            o_io_mem_ar_bits_qos=raw.ar.qos,
            o_io_mem_r_ready=raw.r.ready,
            i_io_mem_r_valid=raw.r.valid,
            i_io_mem_r_bits_data=raw.r.data,
            i_io_mem_r_bits_resp=raw.r.resp,
            i_io_mem_r_bits_last=raw.r.last,
            i_io_mem_r_bits_id=raw.r.id,
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
        if sim_debug:
            ar_seen = Signal()
            self.sync += [
                If(~a.ar.valid, ar_seen.eq(0)),
                If(a.ar.valid & ~ar_seen,
                    ar_seen.eq(1),
                    Display("[vta-axi] ARVALID addr=%08x len=%d ready=%d",
                            a.ar.addr, a.ar.len, a.ar.ready)),
                If(a.ar.valid & a.ar.ready,
                    Display("[vta-axi] AR addr=%08x len=%d size=%d id=%d",
                            a.ar.addr, a.ar.len, a.ar.size, a.ar.id)),
                If(a.r.valid & a.r.ready & a.r.last,
                    Display("[vta-axi] RLAST id=%d resp=%d", a.r.id, a.r.resp)),
                If(a.aw.valid & a.aw.ready,
                    Display("[vta-axi] AW addr=%08x len=%d size=%d id=%d",
                            a.aw.addr, a.aw.len, a.aw.size, a.aw.id)),
                If(a.b.valid & a.b.ready,
                    Display("[vta-axi] B id=%d resp=%d", a.b.id, a.b.resp)),
            ]
