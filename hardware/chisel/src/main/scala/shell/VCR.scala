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
import vta.util.config._
import vta.util.genericbundle._
import vta.interface.axi._
import vta.interface.apb._

/** VCR parameters.
 *
 * These parameters are used on VCR interfaces and modules.
 */
case class VCRParams() {
  val nCtrl = 1
  val nECnt = 1
  val nVals = 1
  val nPtrs = 6
  val nUCnt = 1
  val regBits = 32
}

/** VCRBase. Parametrize base class. */
abstract class VCRBase(implicit p: Parameters) extends GenericParameterizedBundle(p)

/** VCRMaster.
 *
 * This is the master interface used by VCR in the VTAShell to control
 * the Core unit.
 */
class VCRMaster(implicit p: Parameters) extends VCRBase {
  val vp = p(ShellKey).vcrParams
  val mp = p(ShellKey).memParams
  val launch = Output(Bool())
  val finish = Input(Bool())
  val ecnt = Vec(vp.nECnt, Flipped(ValidIO(UInt(vp.regBits.W))))
  val vals = Output(Vec(vp.nVals, UInt(vp.regBits.W)))
  val ptrs = Output(Vec(vp.nPtrs, UInt(mp.addrBits.W)))
  val ucnt = Vec(vp.nUCnt, Flipped(ValidIO(UInt(vp.regBits.W))))
}

/** VCRClient.
 *
 * This is the client interface used by the Core module to communicate
 * to the VCR in the VTAShell.
 */
class VCRClient(implicit p: Parameters) extends VCRBase {
  val vp = p(ShellKey).vcrParams
  val mp = p(ShellKey).memParams
  val launch = Input(Bool())
  val finish = Output(Bool())
  val ecnt = Vec(vp.nECnt, ValidIO(UInt(vp.regBits.W)))
  val vals = Input(Vec(vp.nVals, UInt(vp.regBits.W)))
  val ptrs = Input(Vec(vp.nPtrs, UInt(mp.addrBits.W)))
  val ucnt = Vec(vp.nUCnt, ValidIO(UInt(vp.regBits.W)))
}

/** Protocol-independent request issued by a native VCR bus front-end. */
class VCRWriteRequest(implicit p: Parameters) extends VCRBase {
  val addr = UInt(p(ShellKey).hostParams.addrBits.W)
  val data = UInt(p(ShellKey).vcrParams.regBits.W)
  val strb = UInt((p(ShellKey).vcrParams.regBits / 8).W)
}

/** VTA control-register storage and core-facing behavior. */
class VCRRegisters(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val write = Flipped(ValidIO(new VCRWriteRequest))
    val readAddr = Input(UInt(p(ShellKey).hostParams.addrBits.W))
    val readData = Output(UInt(p(ShellKey).vcrParams.regBits.W))
    val vcr = new VCRMaster
  })

  val vp = p(ShellKey).vcrParams
  val mp = p(ShellKey).memParams

  // registers
  val nPtrs = if (mp.addrBits == 32) vp.nPtrs else 2 * vp.nPtrs
  val nTotal = vp.nCtrl + vp.nECnt + vp.nVals + nPtrs + vp.nUCnt

  val reg = Seq.fill(nTotal)(RegInit(0.U(vp.regBits.W)))
  val addr = Seq.tabulate(nTotal)(_ * 4)
  val reg_map = (addr zip reg) map { case (a, r) => a.U -> r }
  val writeMask = FillInterleaved(8, io.write.bits.strb)
  def mergedWrite(old: UInt): UInt =
    (old & ~writeMask) | (io.write.bits.data & writeMask)
  val eo = vp.nCtrl
  val vo = eo + vp.nECnt
  val po = vo + vp.nVals
  val uo = po + nPtrs

  when(io.vcr.finish) {
    reg(0) := "b_10".U
  }.elsewhen(io.write.valid && addr(0).U === io.write.bits.addr) {
    reg(0) := mergedWrite(reg(0))
  }

  for (i <- 0 until vp.nECnt) {
    when(io.vcr.ecnt(i).valid) {
      reg(eo + i) := io.vcr.ecnt(i).bits
    }.elsewhen(io.write.valid && addr(eo + i).U === io.write.bits.addr) {
      reg(eo + i) := mergedWrite(reg(eo + i))
    }
  }

  for (i <- 0 until (vp.nVals + nPtrs)) {
    when(io.write.valid && addr(vo + i).U === io.write.bits.addr) {
      reg(vo + i) := mergedWrite(reg(vo + i))
    }
  }

  io.readData := MuxLookup(io.readAddr, 0.U, reg_map)

  io.vcr.launch := reg(0)(0)

  for (i <- 0 until vp.nVals) {
    io.vcr.vals(i) := reg(vo + i)
  }

  if (mp.addrBits == 32) { // 32-bit pointers
    for (i <- 0 until nPtrs) {
      io.vcr.ptrs(i) := reg(po + i)
    }
  } else { // 64-bits pointers
    for (i <- 0 until (nPtrs / 2)) {
      io.vcr.ptrs(i) := Cat(reg(po + 2 * i + 1), reg(po + 2 * i))
    }
  }

  for (i <- 0 until vp.nUCnt) {
    when(io.vcr.ucnt(i).valid) {
      reg(uo + i) := io.vcr.ucnt(i).bits
    }.elsewhen(io.write.valid && addr(uo + i).U === io.write.bits.addr) {
      reg(uo + i) := mergedWrite(reg(uo + i))
    }
  }
}

/** Native AXI4-Lite VTA control-register front-end. */
class VCR(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val host = new AXILiteClient(p(ShellKey).hostParams)
    val vcr = new VCRMaster
  })

  val hp = p(ShellKey).hostParams
  val registers = Module(new VCRRegisters)
  val waddr = RegInit("h_ffff".U(hp.addrBits.W))
  val rdata = RegInit(0.U(p(ShellKey).vcrParams.regBits.W))

  val sWriteAddress :: sWriteData :: sWriteResponse :: Nil = Enum(3)
  val wstate = RegInit(sWriteAddress)
  switch(wstate) {
    is(sWriteAddress) { when(io.host.aw.valid) { wstate := sWriteData } }
    is(sWriteData) { when(io.host.w.valid) { wstate := sWriteResponse } }
    is(sWriteResponse) { when(io.host.b.ready) { wstate := sWriteAddress } }
  }
  when(io.host.aw.fire) { waddr := io.host.aw.bits.addr }
  io.host.aw.ready := wstate === sWriteAddress
  io.host.w.ready := wstate === sWriteData
  io.host.b.valid := wstate === sWriteResponse
  io.host.b.bits.resp := 0.U

  val sReadAddress :: sReadData :: Nil = Enum(2)
  val rstate = RegInit(sReadAddress)
  switch(rstate) {
    is(sReadAddress) { when(io.host.ar.valid) { rstate := sReadData } }
    is(sReadData) { when(io.host.r.ready) { rstate := sReadAddress } }
  }
  io.host.ar.ready := rstate === sReadAddress
  io.host.r.valid := rstate === sReadData
  io.host.r.bits.data := rdata
  io.host.r.bits.resp := 0.U

  registers.io.write.valid := io.host.w.fire
  registers.io.write.bits.addr := waddr
  registers.io.write.bits.data := io.host.w.bits.data
  registers.io.write.bits.strb := io.host.w.bits.strb
  registers.io.readAddr := io.host.ar.bits.addr
  when(io.host.ar.fire) { rdata := registers.io.readData }
  io.vcr <> registers.io.vcr
}

/** Native APB4 VTA control-register front-end. */
class VCRAPB(implicit p: Parameters) extends Module {
  private val hp = p(ShellKey).hostParams
  private val ap = APBParams(addrBits = hp.addrBits, dataBits = hp.dataBits)

  val io = IO(new Bundle {
    val host = new APBSlave(ap)
    val vcr = new VCRMaster
  })

  val registers = Module(new VCRRegisters)
  val access = io.host.psel && io.host.penable

  registers.io.write.valid := access && io.host.pwrite
  registers.io.write.bits.addr := io.host.paddr
  registers.io.write.bits.data := io.host.pwdata
  registers.io.write.bits.strb := io.host.pstrb
  registers.io.readAddr := io.host.paddr

  io.host.prdata := registers.io.readData
  io.host.pready := access
  io.host.pslverr := false.B
  io.vcr <> registers.io.vcr
}
