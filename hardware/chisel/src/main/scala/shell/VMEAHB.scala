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

  val io = IO(new Bundle {
    val mem = new AHBMaster(ap)
    val vme = new VMEClient
  })

  val readArb = Module(new Arbiter(new VMECmd, nReadClients))
  for (i <- 0 until nReadClients) {
    readArb.io.in(i) <> io.vme.rd(i).cmd
  }

  val (idle :: readAddress :: readData :: readDeliver :: writeAcquire ::
    writeAddress :: writeData :: Nil) = Enum(7)
  val state = RegInit(idle)
  val address = Reg(UInt(mp.addrBits.W))
  val length = Reg(UInt(mp.lenBits.W))
  val beat = RegInit(0.U(mp.lenBits.W))
  val readClient = Reg(UInt(log2Ceil(nReadClients).W))
  val readTag = Reg(UInt(p(ShellKey).vmeParams.clientTagBitWidth.W))
  val readValue = Reg(UInt(mp.dataBits.W))
  val writeValue = Reg(UInt(mp.dataBits.W))
  val writeAck = RegInit(false.B)

  readArb.io.out.ready := state === idle && !io.vme.wr(0).cmd.valid
  io.vme.wr(0).cmd.ready := state === idle
  io.vme.wr(0).data.ready := state === writeAcquire
  io.vme.wr(0).ack := writeAck
  writeAck := false.B

  for (i <- 0 until nReadClients) {
    io.vme.rd(i).data.valid := state === readDeliver && readClient === i.U
    io.vme.rd(i).data.bits.data := readValue
    io.vme.rd(i).data.bits.tag := readTag
    io.vme.rd(i).data.bits.last := beat === length
  }

  io.mem.haddr := address
  io.mem.hburst := 0.U // Each VME beat is an AHB SINGLE transfer.
  io.mem.hprot := "b0011".U
  io.mem.hsize := ap.sizeConst.U
  io.mem.htrans := Mux(
    state === readAddress || state === writeAddress,
    AHBTransfer.nonseq,
    AHBTransfer.idle)
  io.mem.hwrite := state === writeAddress || state === writeData
  io.mem.hwdata := writeValue

  switch(state) {
    is(idle) {
      beat := 0.U
      when(io.vme.wr(0).cmd.fire) {
        address := io.vme.wr(0).cmd.bits.addr
        length := io.vme.wr(0).cmd.bits.len
        state := writeAcquire
      }.elsewhen(readArb.io.out.fire) {
        address := readArb.io.out.bits.addr
        length := readArb.io.out.bits.len
        readClient := readArb.io.chosen
        readTag := readArb.io.out.bits.tag
        state := readAddress
      }
    }
    is(readAddress) {
      when(io.mem.hready) { state := readData }
    }
    is(readData) {
      when(io.mem.hready) {
        assert(!io.mem.hresp, "AHB read error response")
        readValue := io.mem.hrdata
        state := readDeliver
      }
    }
    is(readDeliver) {
      when(io.vme.rd(readClient).data.ready) {
        when(beat === length) {
          state := idle
        }.otherwise {
          beat := beat + 1.U
          address := address + ap.beatBytes.U
          state := readAddress
        }
      }
    }
    is(writeAcquire) {
      when(io.vme.wr(0).data.fire) {
        assert(io.vme.wr(0).data.bits.strb.andR,
          "AHB VME requires full-width VTA memory writes")
        writeValue := io.vme.wr(0).data.bits.data
        state := writeAddress
      }
    }
    is(writeAddress) {
      when(io.mem.hready) { state := writeData }
    }
    is(writeData) {
      when(io.mem.hready) {
        assert(!io.mem.hresp, "AHB write error response")
        when(beat === length) {
          writeAck := true.B
          state := idle
        }.otherwise {
          beat := beat + 1.U
          address := address + ap.beatBytes.U
          state := writeAcquire
        }
      }
    }
  }
}
