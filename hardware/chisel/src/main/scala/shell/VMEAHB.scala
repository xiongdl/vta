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

package vta.shell

import chisel3._
import chisel3.util._
import vta.interface.ahb._
import vta.util.config._

/** Native AHB-Lite memory engine for the VTA core VME interface. */
class VMEAHB(implicit p: Parameters) extends Module {
  private val mp = p(ShellKey).memParams
  private val ap = AHBParams(addrBits = mp.addrBits, dataBits = mp.dataBits)
  private val nReadClients = p(ShellKey).vmeParams.nReadClients
  require(mp.lenBits <= 4,
    "VMEAHB supports at most 16 beats per VME command")

  val io = IO(new Bundle {
    val mem = new AHBMaster(ap)
    val vme = new VMEClient
  })

  val readArb = Module(new Arbiter(new VMECmd, nReadClients))
  for (i <- 0 until nReadClients) {
    readArb.io.in(i) <> io.vme.rd(i).cmd
  }

  val (idle :: readAddress :: readBurst :: writeAcquire :: writeAddress ::
    writeBurst :: writeWait :: Nil) = Enum(7)
  val state = RegInit(idle)
  val address = Reg(UInt(mp.addrBits.W))
  val length = Reg(UInt(mp.lenBits.W))
  val beat = RegInit(0.U(mp.lenBits.W))
  val burstBeat = RegInit(0.U(4.W))
  val burstLen = RegInit(0.U(4.W))
  val readClient = Reg(UInt(log2Ceil(nReadClients).W))
  val readTag = Reg(UInt(p(ShellKey).vmeParams.clientTagBitWidth.W))
  val writeValue = Reg(UInt(mp.dataBits.W))
  val writeAck = RegInit(false.B)
  val writeQueue = Module(new Queue(new VMEWriteData, 1 << mp.lenBits))
  val readQueue = Module(new Queue(new VMEData, 1 << mp.lenBits))

  def nextBurstLen(beats: UInt, addr: UInt): UInt = {
    // AHB incrementing bursts cannot cross a 1KB address boundary.
    val offset = addr(9, 0)
    val bytesToBoundary = 1024.U(11.W) - offset
    val beatsToBoundary = bytesToBoundary >> log2Ceil(ap.beatBytes)
    val available = Mux(beats < beatsToBoundary, beats, beatsToBoundary)
    Mux(available >= 16.U, 15.U,
      Mux(available >= 8.U, 7.U,
        Mux(available >= 4.U, 3.U, 0.U)))
  }

  writeQueue.io.enq <> io.vme.wr(0).data
  writeQueue.io.deq.ready := state === writeAcquire ||
    (state === writeBurst && io.mem.hready && beat =/= length &&
      burstBeat =/= burstLen) ||
    (state === writeWait && io.mem.hready)

  readArb.io.out.ready := state === idle && readQueue.io.count === 0.U &&
    !io.vme.wr(0).cmd.valid
  io.vme.wr(0).cmd.ready := state === idle && writeQueue.io.count === 0.U
  io.vme.wr(0).ack := writeAck
  writeAck := false.B

  for (i <- 0 until nReadClients) {
    io.vme.rd(i).data.valid := readQueue.io.deq.valid && readClient === i.U
    io.vme.rd(i).data.bits := readQueue.io.deq.bits
  }
  readQueue.io.deq.ready := io.vme.rd(readClient).data.ready
  readQueue.io.enq.valid := state === readBurst && io.mem.hready
  readQueue.io.enq.bits.data := io.mem.hrdata
  readQueue.io.enq.bits.tag := readTag
  readQueue.io.enq.bits.last := beat === length

  io.mem.haddr := Mux(state === readBurst || state === writeBurst,
    address + ap.beatBytes.U, address)
  io.mem.hburst := MuxLookup(burstLen, AHBBurst.single, Seq(
    3.U -> AHBBurst.incr4,
    7.U -> AHBBurst.incr8,
    15.U -> AHBBurst.incr16))
  io.mem.hprot := "b0011".U
  io.mem.hsize := ap.sizeConst.U
  io.mem.htrans := Mux(
    state === readAddress || state === writeAddress,
    AHBTransfer.nonseq,
    Mux(state === readBurst,
      Mux(burstBeat === burstLen, AHBTransfer.idle, AHBTransfer.seq),
      Mux(state === writeBurst,
        Mux(burstBeat === burstLen, AHBTransfer.idle,
          Mux(writeQueue.io.deq.valid, AHBTransfer.seq, AHBTransfer.busy)),
        Mux(state === writeWait,
          Mux(writeQueue.io.deq.valid, AHBTransfer.seq, AHBTransfer.busy),
          AHBTransfer.idle))))
  io.mem.hwrite := state === writeAddress || state === writeBurst || state === writeWait
  io.mem.hwdata := writeValue

  switch(state) {
    is(idle) {
      beat := 0.U
      when(io.vme.wr(0).cmd.fire) {
        assert((io.vme.wr(0).cmd.bits.addr & (ap.beatBytes - 1).U) === 0.U,
          "AHB write address must be aligned to the transfer size")
        address := io.vme.wr(0).cmd.bits.addr
        length := io.vme.wr(0).cmd.bits.len
        burstBeat := 0.U
        burstLen := nextBurstLen(
          io.vme.wr(0).cmd.bits.len +& 1.U, io.vme.wr(0).cmd.bits.addr)
        state := writeAcquire
      }.elsewhen(readArb.io.out.fire) {
        assert((readArb.io.out.bits.addr & (ap.beatBytes - 1).U) === 0.U,
          "AHB read address must be aligned to the transfer size")
        address := readArb.io.out.bits.addr
        length := readArb.io.out.bits.len
        burstBeat := 0.U
        burstLen := nextBurstLen(
          readArb.io.out.bits.len +& 1.U, readArb.io.out.bits.addr)
        readClient := readArb.io.chosen
        readTag := readArb.io.out.bits.tag
        state := readAddress
      }
    }
    is(readAddress) {
      when(io.mem.hready) { state := readBurst }
    }
    is(readBurst) {
      when(io.mem.hready) {
        assert(!io.mem.hresp, "AHB read error response")
        assert(readQueue.io.enq.ready, "AHB read response queue overflow")
        when(beat === length) {
          state := idle
        }.otherwise {
          beat := beat + 1.U
          address := address + ap.beatBytes.U
          when(burstBeat === burstLen) {
            burstBeat := 0.U
            burstLen := nextBurstLen(length - beat, address + ap.beatBytes.U)
            state := readAddress
          }.otherwise {
            burstBeat := burstBeat + 1.U
          }
        }
      }
    }
    is(writeAcquire) {
      when(writeQueue.io.deq.fire) {
        assert(writeQueue.io.deq.bits.strb.andR,
          "AHB VME requires full-width VTA memory writes")
        writeValue := writeQueue.io.deq.bits.data
        state := writeAddress
      }
    }
    is(writeAddress) {
      when(io.mem.hready) { state := writeBurst }
    }
    is(writeBurst) {
      when(io.mem.hready) {
        assert(!io.mem.hresp, "AHB write error response")
        when(beat === length) {
          writeAck := true.B
          state := idle
        }.otherwise {
          beat := beat + 1.U
          address := address + ap.beatBytes.U
          when(burstBeat === burstLen) {
            burstBeat := 0.U
            burstLen := nextBurstLen(length - beat, address + ap.beatBytes.U)
            state := writeAcquire
          }.otherwise {
            burstBeat := burstBeat + 1.U
            when(writeQueue.io.deq.valid) {
              assert(writeQueue.io.deq.bits.strb.andR,
                "AHB VME requires full-width VTA memory writes")
              writeValue := writeQueue.io.deq.bits.data
            }.otherwise {
              state := writeWait
            }
          }
        }
      }
    }
    is(writeWait) {
      when(io.mem.hready && writeQueue.io.deq.valid) {
        assert(writeQueue.io.deq.bits.strb.andR,
          "AHB VME requires full-width VTA memory writes")
        writeValue := writeQueue.io.deq.bits.data
        state := writeBurst
      }
    }
  }
}
