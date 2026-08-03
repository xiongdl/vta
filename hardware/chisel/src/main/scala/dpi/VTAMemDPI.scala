/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package vta.dpi

import chisel3._
import chisel3.util._
import chisel3.experimental.IntParam
import vta.util.config._
import vta.interface.axi._
import vta.interface.ahb._
import vta.shell._

/** Memory DPI parameters */
case class VTAMemDPIParams(
    dpiDelay  : Int,
    dpiLenBits: Int,
    dpiAddrBits: Int,
    dpiDataBits: Int,
    dpiTagBits: Int
) {}
case object DpiKey extends Field[VTAMemDPIParams]
case object TSIMMemReadDelayKey extends Field[Int](
  sys.env.get("VTA_TSIM_MEM_READ_LATENCY").map(_.toInt).getOrElse(0))

/** Common delay applied before issuing a TSIM DPI memory read request. */
class VTAMemDPIReadDelay(delay: Int) extends Module {
  require(delay >= 0, "TSIM memory read delay must be non-negative")
  val io = IO(new Bundle {
    val start = Input(Bool())
    val issue = Output(Bool())
  })
  private val countBits = math.max(1, log2Ceil(delay + 1))
  val count = RegInit(0.U(countBits.W))
  val active = RegInit(false.B)
  if (delay > 0) {
    when(io.start) {
      count := delay.U
      active := true.B
    }.elsewhen(active && count === 1.U) {
      count := 0.U
      active := false.B
    }.elsewhen(active) {
      count := count - 1.U
    }
  }
  io.issue := (if (delay == 0) io.start else active && count === 1.U)
}

/** Memory master interface.
 *
 * This interface is tipically used by the Accelerator
 */

class MemRequest(implicit val p: Parameters) extends Bundle {
  val len =   (UInt(p(ShellKey).memParams.dataBits.W))
  val addr = (UInt(p(ShellKey).memParams.addrBits.W))
  val id = (UInt(p(ShellKey).memParams.idBits.W))
}

class VTAMemDPIData(implicit val p: Parameters) extends Bundle {
  val data = UInt(p(ShellKey).memParams.dataBits.W)
  val id   = UInt(p(ShellKey).memParams.idBits.W)
}

class VTAMemDPIWrData(implicit val p: Parameters) extends Bundle {
  val data = UInt(p(ShellKey).memParams.dataBits.W)
  val strb = UInt((p(ShellKey).memParams.dataBits/8).W)
}


class VTAMemDPIMaster(implicit val p: Parameters) extends Bundle {
  val req = new Bundle {
    val ar_valid = Output(Bool())
    val ar_len =   Output(UInt(p(ShellKey).memParams.lenBits.W))
    val ar_addr = Output(UInt(p(ShellKey).memParams.addrBits.W))
    val ar_id   = Output(UInt(p(ShellKey).memParams.idBits.W))
    val aw_valid = Output(Bool())
    val aw_addr = Output(UInt(p(ShellKey).memParams.addrBits.W))
    val aw_len  = Output(UInt(p(ShellKey).memParams.lenBits.W))
  }
  val wr = ValidIO(new VTAMemDPIWrData)
  val rd = Flipped(Decoupled(new VTAMemDPIData))
}

/** Memory client interface.
 *
 * This interface is tipically used by the Host
 */
class VTAMemDPIClient(implicit val p: Parameters) extends Bundle {
  val req = new Bundle {
    val ar_valid = Input(Bool())
    val ar_len =   Input(UInt(p(ShellKey).memParams.lenBits.W))
    val ar_addr = Input(UInt(p(ShellKey).memParams.addrBits.W))
    val ar_id   = Input(UInt(p(ShellKey).memParams.idBits.W))
    val aw_valid = Input(Bool())
    val aw_addr = Input(UInt(p(ShellKey).memParams.addrBits.W))
    val aw_len  = Input(UInt(p(ShellKey).memParams.lenBits.W))
  }
  val wr = Flipped(ValidIO(new VTAMemDPIWrData))
  val rd = (Decoupled(new VTAMemDPIData))
}

/** Memory DPI module.
 *
 * Wrapper for Memory Verilog DPI module.
 */
class VTAMemDPI(implicit val p: Parameters) extends BlackBox(
  Map(
    "LEN_BITS" -> IntParam(p(ShellKey).memParams.lenBits),
    "ADDR_BITS" -> IntParam(p(ShellKey).memParams.addrBits),
    "DATA_BITS" -> IntParam(p(ShellKey).memParams.dataBits))) with HasBlackBoxResource {

  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Reset())
    val dpi = new VTAMemDPIClient
  })
  addResource("/verilog/VTAMemDPI.v")
}

class VTAMemDPIToAXI(debug: Boolean = true)(implicit val p: Parameters) extends Module {
  val io = IO(new Bundle {
    val dpi = new VTAMemDPIMaster
    val axi = new AXIClient(p(ShellKey).memParams)
  })
  val readDelay = Module(new VTAMemDPIReadDelay(p(TSIMMemReadDelayKey)))
  //Read request interface for sw memory manager
  val ar_valid = RegInit(false.B)
  val ar_len = RegInit(0.U.asTypeOf(chiselTypeOf(io.dpi.req.ar_len)))
  val ar_addr = RegInit(0.U.asTypeOf(chiselTypeOf(io.dpi.req.ar_addr)))
  val ar_id   = RegInit(0.U.asTypeOf(chiselTypeOf(io.dpi.req.ar_len)))
  val rd_data = RegInit(0.U.asTypeOf(chiselTypeOf(io.dpi.rd.bits.data)))
  val rIdle :: readData :: Nil = Enum(2)
  val rstate = RegInit(rIdle)
  //Write request interface for sw memomry manager
  val aw_valid = RegInit(false.B)
  val aw_len = RegInit(0.U.asTypeOf(chiselTypeOf(io.dpi.req.aw_len)))
  val aw_addr = RegInit(0.U.asTypeOf(chiselTypeOf(io.dpi.req.aw_addr)))
  val wIdle :: writeAddress :: writeData :: writeResponse :: Nil = Enum(4)
  val wstate = RegInit(wIdle)
  //Read Interface to Memory Manager
  val dpiReqQueue = Module(new Queue(new MemRequest, 256))
  dpiReqQueue.io.enq.valid     := io.axi.ar.valid
  dpiReqQueue.io.enq.bits.addr := io.axi.ar.bits.addr
  dpiReqQueue.io.enq.bits.len  := io.axi.ar.bits.len
  dpiReqQueue.io.enq.bits.id   := io.axi.ar.bits.id

  switch(rstate){
    is(rIdle){
      when(dpiReqQueue.io.deq.fire) {
        rstate := readData
      }
    }
    is(readData) {
      when(io.axi.r.ready && io.dpi.rd.valid && ar_len === 0.U) {
        rstate := rIdle
      }
    }
  }
  when(rstate === rIdle) {
    when(dpiReqQueue.io.deq.fire) {
      ar_len :=  dpiReqQueue.io.deq.bits.len
      ar_addr := dpiReqQueue.io.deq.bits.addr
      ar_id   := dpiReqQueue.io.deq.bits.id
    }
  }
  .elsewhen(rstate === readData){
    when(io.axi.r.ready && io.dpi.rd.valid && ar_len =/= 0.U){
      ar_len := ar_len - 1.U
    }
  }
dpiReqQueue.io.deq.ready := rstate === rIdle
when(dpiReqQueue.io.deq.fire) {
  io.dpi.req.ar_len  := dpiReqQueue.io.deq.bits.len
  io.dpi.req.ar_addr := dpiReqQueue.io.deq.bits.addr
  io.dpi.req.ar_id   := dpiReqQueue.io.deq.bits.id
  io.dpi.req.ar_valid := readDelay.io.issue
  }.otherwise{
    io.dpi.req.ar_len  := ar_len
    io.dpi.req.ar_addr := ar_addr
    io.dpi.req.ar_id   := ar_id
    io.dpi.req.ar_valid := readDelay.io.issue
  }
  io.axi.ar.ready := dpiReqQueue.io.enq.ready
  io.axi.r.valid := io.dpi.rd.valid
  io.axi.r.bits.data := io.dpi.rd.bits.data
  io.axi.r.bits.last := (ar_len === 0.U && io.dpi.rd.valid)
  io.axi.r.bits.resp := 0.U
  io.axi.r.bits.user := 0.U
  io.axi.r.bits.id := io.dpi.rd.bits.id
  io.dpi.rd.ready := io.axi.r.ready
  readDelay.io.start := dpiReqQueue.io.deq.fire

  //Write Request
  switch(wstate){
    is(wIdle){
      when(io.axi.aw.valid){
        wstate := writeAddress
      }
    }
    is(writeAddress) {
      when(io.axi.aw.valid) {
        wstate := writeData
      }
    }
    is(writeData) {
      when(io.axi.w.valid && io.axi.w.bits.last) {
        wstate := writeResponse
      }
    }
    is(writeResponse) {
      when(io.axi.b.ready) {
        wstate := wIdle
      }
    }
  }
  when(wstate === wIdle){
    when(io.axi.aw.valid){
      aw_len := io.axi.aw.bits.len
      aw_addr := io.axi.aw.bits.addr
    }
  }
  io.dpi.req.aw_addr := aw_addr
  io.dpi.req.aw_len  := aw_len
  io.dpi.req.aw_valid := RegNext(io.axi.aw.valid) & (wstate === writeAddress)
  io.axi.aw.ready := wstate === writeAddress
  io.dpi.wr.valid := wstate === writeData & io.axi.w.valid
  io.dpi.wr.bits.data := io.axi.w.bits.data
  io.dpi.wr.bits.strb := io.axi.w.bits.strb
  io.axi.w.ready := wstate === writeData

  io.axi.b.valid := wstate === writeResponse
  io.axi.b.bits.resp := 0.U
  io.axi.b.bits.user := 0.U
  io.axi.b.bits.id := 0.U
}

/** Native AHB-Lite slave backed directly by the memory DPI interface. */
class VTAMemDPIToAHB(debug: Boolean = false)(implicit val p: Parameters) extends Module {
  private val mp = p(ShellKey).memParams
  private val ap = AHBParams(addrBits = mp.addrBits, dataBits = mp.dataBits)

  val io = IO(new Bundle {
    val dpi = new VTAMemDPIMaster
    val ahb = new AHBSlave(ap)
  })
  val readDelay = Module(new VTAMemDPIReadDelay(p(TSIMMemReadDelayKey)))

  val idle :: readData :: writeData :: writeWait :: Nil = Enum(4)
  val state = RegInit(idle)
  val transfer = io.ahb.htrans(1)
  val readSetup = state === idle && transfer && !io.ahb.hwrite
  val writeSetup = state === idle && transfer && io.ahb.hwrite
  val burstLen = RegInit(0.U(4.W))
  val burstBeat = RegInit(0.U(4.W))
  val readAddr = Reg(UInt(mp.addrBits.W))
  val readRequestLen = RegInit(0.U(4.W))
  val undefinedBurst = RegInit(false.B)

  val requestLen = MuxLookup(io.ahb.hburst, 0.U, Seq(
    AHBBurst.incr4 -> 3.U,
    AHBBurst.incr8 -> 7.U,
    AHBBurst.incr16 -> 15.U))
  val readContinue = state === readData && undefinedBurst && io.dpi.rd.fire &&
    io.ahb.htrans === AHBTransfer.seq && !io.ahb.hwrite
  val readStart = readSetup || readContinue
  val writeContinue = state === writeData && undefinedBurst &&
    io.ahb.htrans === AHBTransfer.seq && io.ahb.hwrite
  val writeResume = state === writeWait && undefinedBurst &&
    io.ahb.htrans === AHBTransfer.seq && io.ahb.hwrite
  val writeStart = writeSetup || writeContinue || writeResume

  // Fixed-length bursts map to one DPI burst request.  Each beat of an
  // undefined-length INCR maps to a DPI SINGLE because AHB carries no length.
  io.dpi.req.ar_valid := readDelay.io.issue
  io.dpi.req.ar_len := Mux(readStart, requestLen, readRequestLen)
  io.dpi.req.ar_addr := Mux(readStart, io.ahb.haddr, readAddr)
  io.dpi.req.ar_id := 0.U
  io.dpi.req.aw_valid := writeStart
  io.dpi.req.aw_len := requestLen
  io.dpi.req.aw_addr := io.ahb.haddr

  io.dpi.wr.valid := state === writeData
  io.dpi.wr.bits.data := io.ahb.hwdata
  io.dpi.wr.bits.strb := Fill(mp.strbBits, true.B)
  // VTAMemDPI registers the C++ response.  Advancing the DPI memory on the
  // request cycle prepares the first beat for the following AHB data phase.
  readDelay.io.start := readStart
  io.dpi.rd.ready := state === readData || readSetup

  io.ahb.hrdata := io.dpi.rd.bits.data
  io.ahb.hready := state === idle || state === writeData || state === writeWait ||
    (state === readData && io.dpi.rd.valid)
  io.ahb.hresp := false.B

  when(readStart) {
    readAddr := io.ahb.haddr
    readRequestLen := requestLen
  }

  when(readSetup) {
    burstLen := requestLen
    burstBeat := 0.U
    undefinedBurst := io.ahb.hburst === AHBBurst.incr
    state := readData
  }.elsewhen(writeSetup) {
    burstLen := requestLen
    burstBeat := 0.U
    undefinedBurst := io.ahb.hburst === AHBBurst.incr
    state := writeData
  }.elsewhen(state === readData && io.dpi.rd.fire) {
    when(undefinedBurst) {
      when(readContinue) {
        burstBeat := burstBeat + 1.U
      }.otherwise {
        state := idle
      }
    }.otherwise {
      when(burstBeat === burstLen) {
        state := idle
      }.otherwise {
        assert(io.ahb.htrans === AHBTransfer.seq && !io.ahb.hwrite,
          "AHB read burst requires a sequential read transfer")
        burstBeat := burstBeat + 1.U
      }
    }
  }.elsewhen(state === writeData) {
    when(undefinedBurst) {
      when(io.ahb.htrans === AHBTransfer.busy) {
        state := writeWait
      }.elsewhen(writeContinue) {
        burstBeat := burstBeat + 1.U
      }.otherwise {
        state := idle
      }
    }.otherwise {
      when(burstBeat === burstLen) {
        state := idle
      }.otherwise {
        burstBeat := burstBeat + 1.U
        when(io.ahb.htrans === AHBTransfer.busy) {
          state := writeWait
        }.otherwise {
          assert(io.ahb.htrans === AHBTransfer.seq && io.ahb.hwrite,
            "AHB write burst requires a sequential or busy transfer")
        }
      }
    }
  }.elsewhen(state === writeWait) {
    when(io.ahb.htrans === AHBTransfer.seq) {
      assert(io.ahb.hwrite, "AHB write burst requires a write transfer")
      state := writeData
    }.otherwise {
      assert(io.ahb.htrans === AHBTransfer.busy,
        "AHB write burst wait requires a busy or sequential transfer")
    }
  }

  if (debug) {
    when(readSetup) {
      printf("[VTAMemDPIToAHB] read addr:%x\n", io.ahb.haddr)
    }
    when(writeSetup) {
      printf("[VTAMemDPIToAHB] write addr:%x\n", io.ahb.haddr)
    }
    when(state === writeData) {
      printf("[VTAMemDPIToAHB] write data:%x\n", io.ahb.hwdata)
    }
  }
}
