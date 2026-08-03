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
import vta.util.config._
import vta.interface.axi._
import vta.interface.apb._
import vta.shell._

/** Host DPI parameters */
trait VTAHostDPIParams {
  val dpiAddrBits = 8
  val dpiDataBits = 32
}

/** Host master interface.
 *
 * This interface is tipically used by the Host
 */
class VTAHostDPIMaster extends Bundle with VTAHostDPIParams {
  val req = new Bundle {
    val valid = Output(Bool())
    val opcode = Output(Bool())
    val addr = Output(UInt(dpiAddrBits.W))
    val value = Output(UInt(dpiDataBits.W))
    val deq = Input(Bool())
  }
  val resp = Flipped(ValidIO(UInt(dpiDataBits.W)))
}

/** Host client interface.
 *
 * This interface is tipically used by the Accelerator
 */
class VTAHostDPIClient extends Bundle with VTAHostDPIParams {
  val req = new Bundle {
    val valid = Input(Bool())
    val opcode = Input(Bool())
    val addr = Input(UInt(dpiAddrBits.W))
    val value = Input(UInt(dpiDataBits.W))
    val deq = Output(Bool())
  }
  val resp = ValidIO(UInt(dpiDataBits.W))
}

/** Host DPI module.
 *
 * Wrapper for Host Verilog DPI module.
 */
class VTAHostDPI extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Reset())
    val dpi = new VTAHostDPIMaster
  })
  addResource("/verilog/VTAHostDPI.v")
}

/** Host DPI to AXI Converter.
 *
 * Convert Host DPI to AXI for VTAShell
 */
class VTAHostDPIToAXI(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val dpi = new VTAHostDPIClient
    val axi = new AXILiteMaster(p(ShellKey).hostParams)
  })
  val addr = RegInit(0.U.asTypeOf(chiselTypeOf(io.dpi.req.addr)))
  val data = RegInit(0.U.asTypeOf(chiselTypeOf(io.dpi.req.value)))
  val sIdle :: sReadAddress :: sReadData :: sWriteAddress :: sWriteData :: sWriteResponse :: Nil =
    Enum(6)
  val state = RegInit(sIdle)

  switch(state) {
    is(sIdle) {
      when(io.dpi.req.valid) {
        when(io.dpi.req.opcode) {
          state := sWriteAddress
        }.otherwise {
          state := sReadAddress
        }
      }
    }
    is(sReadAddress) {
      when(io.axi.ar.ready) {
        state := sReadData
      }
    }
    is(sReadData) {
      when(io.axi.r.valid) {
        state := sIdle
      }
    }
    is(sWriteAddress) {
      when(io.axi.aw.ready) {
        state := sWriteData
      }
    }
    is(sWriteData) {
      when(io.axi.w.ready) {
        state := sWriteResponse
      }
    }
    is(sWriteResponse) {
      when(io.axi.b.valid) {
        state := sIdle
      }
    }
  }

  when(state === sIdle && io.dpi.req.valid) {
    addr := io.dpi.req.addr
    data := io.dpi.req.value
  }

  io.axi.aw.valid := state === sWriteAddress
  io.axi.aw.bits.addr := addr
  io.axi.w.valid := state === sWriteData
  io.axi.w.bits.data := data
  io.axi.w.bits.strb := "h_f".U
  io.axi.b.ready := state === sWriteResponse

  io.axi.ar.valid := state === sReadAddress
  io.axi.ar.bits.addr := addr
  io.axi.r.ready := state === sReadData

  io.dpi.req.deq := (state === sReadAddress & io.axi.ar.ready) | (state === sWriteAddress & io.axi.aw.ready)
  io.dpi.resp.valid := io.axi.r.valid
  io.dpi.resp.bits := io.axi.r.bits.data

  if (debug) {
    when(state === sWriteAddress && io.axi.aw.ready) {
      printf("[VTAHostDPIToAXI] [AW] addr:%x\n", addr)
    }
    when(state === sReadAddress && io.axi.ar.ready) {
      printf("[VTAHostDPIToAXI] [AR] addr:%x\n", addr)
    }
    when(io.axi.r.fire) {
      printf("[VTAHostDPIToAXI] [R] value:%x\n", io.axi.r.bits.data)
    }
    when(io.axi.w.fire) {
      printf("[VTAHostDPIToAXI] [W] value:%x\n", io.axi.w.bits.data)
    }
  }
}

/** Host DPI to APB4 converter. */
class VTAHostDPIToAPB(debug: Boolean = false)(implicit p: Parameters) extends Module {
  private val hp = p(ShellKey).hostParams
  private val ap = APBParams(addrBits = hp.addrBits, dataBits = hp.dataBits)

  val io = IO(new Bundle {
    val dpi = new VTAHostDPIClient
    val apb = new APBMaster(ap)
  })

  val idle :: access :: Nil = Enum(2)
  val state = RegInit(idle)
  val addr = Reg(UInt(ap.addrBits.W))
  val data = Reg(UInt(ap.dataBits.W))
  val write = Reg(Bool())

  // Drive the APB setup phase directly from the DPI request.  The request is
  // captured at the setup/access boundary and remains stable through access.
  val setup = state === idle && io.dpi.req.valid
  io.apb.psel := setup || state === access
  io.apb.penable := state === access
  io.apb.paddr := Mux(setup, io.dpi.req.addr, addr)
  io.apb.pwrite := Mux(setup, io.dpi.req.opcode, write)
  io.apb.pwdata := Mux(setup, io.dpi.req.value, data)
  io.apb.pstrb := Fill(ap.strbBits, true.B)
  io.apb.pprot := 0.U

  when(setup) {
    addr := io.dpi.req.addr
    data := io.dpi.req.value
    write := io.dpi.req.opcode
    state := access
  }.elsewhen(state === access && io.apb.pready) {
    state := idle
  }

  val complete = state === access && io.apb.pready
  // Dequeue at the setup/access boundary.  VTAHostDPI registers its outputs,
  // so this also gives it the access phase to present the next request before
  // the converter returns to idle.
  io.dpi.req.deq := setup
  io.dpi.resp.valid := complete && !write
  io.dpi.resp.bits := io.apb.prdata

  if (debug) {
    when(complete) {
      printf("[VTAHostDPIToAPB] write:%x addr:%x data:%x error:%x\n",
        write, addr, Mux(write, data, io.apb.prdata), io.apb.pslverr)
    }
  }
}
