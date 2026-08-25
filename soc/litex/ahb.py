"""AHB-Lite adapters used by the VTA-only SoC paths."""

from migen import FSM, If, Module, NextState, NextValue, Signal


class AHBInterface:
    """Minimal signal bundle matching the frozen Chisel AHB master."""

    def __init__(self, data_width, address_width=32):
        self.data_width = data_width
        self.addr = Signal(address_width)
        self.burst = Signal(3)
        self.prot = Signal(4)
        self.size = Signal(3)
        self.trans = Signal(2)
        self.wdata = Signal(data_width)
        self.write = Signal()
        self.rdata = Signal(data_width)
        self.ready = Signal()
        self.resp = Signal()


class LiteDRAMAHB2Native(Module):
    """Serialize AHB-Lite transfers onto a matching-width LiteDRAM port.

    AHB bursts remain visible to the frozen master, but each accepted beat is
    issued as one native command.  Holding HREADY low preserves the AHB address
    and data phases while LiteDRAM applies back-pressure.
    """

    def __init__(self, ahb, port, base_address):
        if ahb.data_width != port.data_width:
            raise ValueError("AHB and LiteDRAM native widths must match")
        address_shift = (ahb.data_width // 8).bit_length() - 1
        address = Signal.like(ahb.addr)
        read_data = Signal.like(ahb.rdata)
        write_data = Signal.like(ahb.wdata)
        cmd_done = Signal()
        data_done = Signal()
        transfer = ahb.trans[1]

        self.comb += [
            ahb.ready.eq(0),
            ahb.resp.eq(0),
            ahb.rdata.eq(read_data),
            port.cmd.valid.eq(0),
            port.cmd.we.eq(0),
            port.cmd.last.eq(1),
            port.cmd.addr.eq((address - base_address) >> address_shift),
            port.wdata.valid.eq(0),
            port.wdata.data.eq(write_data),
            port.wdata.we.eq((1 << (ahb.data_width // 8)) - 1),
            port.rdata.ready.eq(0),
        ]

        self.submodules.fsm = fsm = FSM(reset_state="IDLE")

        def accept_address():
            return If(transfer,
                NextValue(address, ahb.addr),
                If(ahb.write, NextState("WRITE_DATA")).Else(NextState("READ_CMD")))

        fsm.act("IDLE", ahb.ready.eq(1), accept_address())
        # HWDATA belongs to the cycle after the accepted write address phase.
        fsm.act("WRITE_DATA",
            NextValue(write_data, ahb.wdata),
            NextValue(cmd_done, 0),
            NextValue(data_done, 0),
            NextState("WRITE_SEND"))
        fsm.act("WRITE_SEND",
            port.cmd.valid.eq(~cmd_done),
            port.cmd.we.eq(1),
            port.wdata.valid.eq(~data_done),
            If(port.cmd.valid & port.cmd.ready, NextValue(cmd_done, 1)),
            If(port.wdata.valid & port.wdata.ready, NextValue(data_done, 1)),
            If((cmd_done | (port.cmd.valid & port.cmd.ready)) &
               (data_done | (port.wdata.valid & port.wdata.ready)),
                NextState("COMPLETE")))
        fsm.act("READ_CMD",
            port.cmd.valid.eq(1),
            If(port.cmd.ready, NextState("READ_DATA")))
        fsm.act("READ_DATA",
            port.rdata.ready.eq(1),
            If(port.rdata.valid,
                NextValue(read_data, port.rdata.data),
                NextState("COMPLETE")))
        # COMPLETE finishes the prior data phase and accepts the next pipelined
        # address phase in the same cycle when one is present.
        fsm.act("COMPLETE",
            ahb.ready.eq(1),
            If(transfer, accept_address()).Else(NextState("IDLE")))
