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
import vta.interface.axi._
import vta.interface.apb._
import vta.interface.ahb._
import vta.shell._
import vta.dpi._

/** Simulation-only AXI4-Lite to APB4 host adapter. */
class SimAXILiteToAPB(implicit p: Parameters) extends Module {
  private val hp = p(ShellKey).hostParams
  private val ap = APBParams(addrBits = hp.addrBits, dataBits = hp.dataBits)

  val io = IO(new Bundle {
    val axi = new AXILiteClient(hp)
    val apb = new APBMaster(ap)
  })

  val (idle :: writeData :: writeSetup :: writeAccess :: writeResponse ::
    readSetup :: readAccess :: readResponse :: Nil) = Enum(8)
  val state = RegInit(idle)
  val addr = Reg(UInt(hp.addrBits.W))
  val data = Reg(UInt(hp.dataBits.W))
  val strb = Reg(UInt(hp.strbBits.W))
  val response = RegInit(0.U(hp.respBits.W))

  io.axi.aw.ready := state === idle
  io.axi.w.ready := state === writeData
  io.axi.b.valid := state === writeResponse
  io.axi.b.bits.resp := response
  io.axi.ar.ready := state === idle && !io.axi.aw.valid
  io.axi.r.valid := state === readResponse
  io.axi.r.bits.data := data
  io.axi.r.bits.resp := response

  io.apb.paddr := addr
  io.apb.psel := state === writeSetup || state === writeAccess ||
    state === readSetup || state === readAccess
  io.apb.penable := state === writeAccess || state === readAccess
  io.apb.pwrite := state === writeSetup || state === writeAccess
  io.apb.pwdata := data
  io.apb.pstrb := strb
  io.apb.pprot := 0.U

  switch(state) {
    is(idle) {
      when(io.axi.aw.fire) {
        addr := io.axi.aw.bits.addr
        state := writeData
      }.elsewhen(io.axi.ar.fire) {
        addr := io.axi.ar.bits.addr
        state := readSetup
      }
    }
    is(writeData) {
      when(io.axi.w.fire) {
        data := io.axi.w.bits.data
        strb := io.axi.w.bits.strb
        state := writeSetup
      }
    }
    is(writeSetup) { state := writeAccess }
    is(writeAccess) {
      when(io.apb.pready) {
        response := Mux(io.apb.pslverr, 2.U, 0.U)
        state := writeResponse
      }
    }
    is(writeResponse) { when(io.axi.b.ready) { state := idle } }
    is(readSetup) { state := readAccess }
    is(readAccess) {
      when(io.apb.pready) {
        data := io.apb.prdata
        response := Mux(io.apb.pslverr, 2.U, 0.U)
        state := readResponse
      }
    }
    is(readResponse) { when(io.axi.r.ready) { state := idle } }
  }
}

/** VTAHost.
 *
 * This module translate the DPI protocol into AXI. This is a simulation only
 * module and used to test host-to-VTA communication. This module should be updated
 * for testing hosts using a different bus protocol, other than AXI.
 */
class VTAHost(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val axi = new AXILiteMaster(p(ShellKey).hostParams)
  })
  val host_dpi = Module(new VTAHostDPI)
  val host_axi = Module(new VTAHostDPIToAXI)
  host_dpi.io.reset := reset
  host_dpi.io.clock := clock
  host_axi.io.dpi <> host_dpi.io.dpi
  io.axi <> host_axi.io.axi
}

/** VTAMem.
 *
 * This module translate the DPI protocol into AXI. This is a simulation only
 * module and used to test VTA-to-memory communication. This module should be updated
 * for testing memories using a different bus protocol, other than AXI.
 */
class VTAMem(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val axi = new AXIClient(p(ShellKey).memParams)
  })
  val mem_dpi = Module(new VTAMemDPI)
  val mem_axi = Module(new VTAMemDPIToAXI)
  mem_dpi.io.reset := reset
  mem_dpi.io.clock := clock
  mem_dpi.io.dpi <> mem_axi.io.dpi
  mem_axi.io.axi <> io.axi
}

/** VTASim.
 *
 * This module is used to handle hardware simulation thread, such as halting
 * or terminating the simulation thread. The sim_wait port is used to halt
 * the simulation thread when it is asserted and resume it when it is
 * de-asserted.
 */
class VTASim(implicit p: Parameters) extends Module {
  val sim_wait = IO(Output(Bool()))
  val sim = Module(new VTASimDPI)
  sim.io.reset := reset
  sim.io.clock := clock
  sim_wait := sim.io.dpi_wait
}

/** SimShell.
 *
 * The simulation shell instantiate the sim, host and memory DPI modules that
 * are connected to the VTAShell. An extra clock, sim_clock, is used to eval
 * the VTASim DPI function when the main simulation clock is on halt state.
 */
class SimShell(implicit p: Parameters) extends Module {
  val mem = IO(new AXIClient(p(ShellKey).memParams))
  val host = IO(new AXILiteMaster(p(ShellKey).hostParams))
  val sim_clock = IO(Input(Clock()))
  val sim_wait = IO(Output(Bool()))
  val mod_sim = Module(new VTASim)
  val mod_host = Module(new VTAHost)
  val mod_mem = Module(new VTAMem)
  mem <> mod_mem.io.axi
  host <> mod_host.io.axi
  mod_sim.reset := reset
  mod_sim.clock := sim_clock
  sim_wait := mod_sim.sim_wait
}

/** APB-host simulation shell. C++ DPI remains on its existing AXI interface. */
class SimShellAPB(implicit p: Parameters) extends Module {
  private val hp = p(ShellKey).hostParams
  private val ap = APBParams(addrBits = hp.addrBits, dataBits = hp.dataBits)

  val mem = IO(new AXIClient(p(ShellKey).memParams))
  val host = IO(new APBMaster(ap))
  val sim_clock = IO(Input(Clock()))
  val sim_wait = IO(Output(Bool()))

  val modSim = Module(new VTASim)
  val modHost = Module(new VTAHost)
  val hostAdapter = Module(new SimAXILiteToAPB)
  val modMem = Module(new VTAMem)

  mem <> modMem.io.axi
  modHost.io.axi <> hostAdapter.io.axi
  host <> hostAdapter.io.apb
  modSim.reset := reset
  modSim.clock := sim_clock
  sim_wait := modSim.sim_wait
}

/** Simulation-only AHB-Lite slave to AXI4 memory adapter. */
class SimAHBToAXI(implicit p: Parameters) extends Module {
  private val mp = p(ShellKey).memParams
  private val ap = AHBParams(addrBits = mp.addrBits, dataBits = mp.dataBits)

  val io = IO(new Bundle {
    val ahb = new AHBSlave(ap)
    val axi = new AXIMaster(mp)
  })

  val (idle :: writeCapture :: writeAddress :: writeData :: writeResponse ::
    readAddress :: readData :: Nil) = Enum(7)
  val state = RegInit(idle)
  val address = Reg(UInt(mp.addrBits.W))
  val data = Reg(UInt(mp.dataBits.W))

  io.ahb.hrdata := Mux(state === readData, io.axi.r.bits.data, 0.U)
  io.ahb.hready := state === idle ||
    (state === writeResponse && io.axi.b.valid) ||
    (state === readData && io.axi.r.valid)
  io.ahb.hresp := Mux(
    state === writeResponse,
    io.axi.b.bits.resp =/= 0.U,
    state === readData && io.axi.r.bits.resp =/= 0.U)

  io.axi.aw.valid := state === writeAddress
  io.axi.aw.bits.addr := address
  io.axi.aw.bits.id := 0.U
  io.axi.aw.bits.len := 0.U
  io.axi.w.valid := state === writeData
  io.axi.w.bits.data := data
  io.axi.w.bits.strb := Fill(mp.strbBits, true.B)
  io.axi.w.bits.last := true.B
  io.axi.w.bits.id := 0.U
  io.axi.b.ready := state === writeResponse
  io.axi.ar.valid := state === readAddress
  io.axi.ar.bits.addr := address
  io.axi.ar.bits.id := 0.U
  io.axi.ar.bits.len := 0.U
  io.axi.r.ready := state === readData
  io.axi.setConst()

  switch(state) {
    is(idle) {
      when(io.ahb.htrans(1)) {
        address := io.ahb.haddr
        when(io.ahb.hwrite) {
          state := writeCapture
        }.otherwise {
          state := readAddress
        }
      }
    }
    is(writeCapture) {
      data := io.ahb.hwdata
      state := writeAddress
    }
    is(writeAddress) { when(io.axi.aw.fire) { state := writeData } }
    is(writeData) { when(io.axi.w.fire) { state := writeResponse } }
    is(writeResponse) { when(io.axi.b.fire) { state := idle } }
    is(readAddress) { when(io.axi.ar.fire) { state := readData } }
    is(readData) { when(io.axi.r.fire) { state := idle } }
  }
}

/** AHB-memory simulation shell with the unchanged AXI C++ DPI ABI. */
class SimShellAHB(implicit p: Parameters) extends Module {
  private val mp = p(ShellKey).memParams
  private val ap = AHBParams(addrBits = mp.addrBits, dataBits = mp.dataBits)

  val mem = IO(new AHBSlave(ap))
  val host = IO(new AXILiteMaster(p(ShellKey).hostParams))
  val sim_clock = IO(Input(Clock()))
  val sim_wait = IO(Output(Bool()))

  val modSim = Module(new VTASim)
  val modHost = Module(new VTAHost)
  val memAdapter = Module(new SimAHBToAXI)
  val modMem = Module(new VTAMem)

  host <> modHost.io.axi
  mem <> memAdapter.io.ahb
  memAdapter.io.axi <> modMem.io.axi
  modSim.reset := reset
  modSim.clock := sim_clock
  sim_wait := modSim.sim_wait
}

/** APB-host/AHB-memory simulation shell; both adapters are simulation-only. */
class SimShellAPBAHB(implicit p: Parameters) extends Module {
  private val hp = p(ShellKey).hostParams
  private val mp = p(ShellKey).memParams
  private val hostAP = APBParams(addrBits = hp.addrBits, dataBits = hp.dataBits)
  private val memAP = AHBParams(addrBits = mp.addrBits, dataBits = mp.dataBits)

  val mem = IO(new AHBSlave(memAP))
  val host = IO(new APBMaster(hostAP))
  val sim_clock = IO(Input(Clock()))
  val sim_wait = IO(Output(Bool()))

  val modSim = Module(new VTASim)
  val modHost = Module(new VTAHost)
  val hostAdapter = Module(new SimAXILiteToAPB)
  val memAdapter = Module(new SimAHBToAXI)
  val modMem = Module(new VTAMem)

  modHost.io.axi <> hostAdapter.io.axi
  host <> hostAdapter.io.apb
  mem <> memAdapter.io.ahb
  memAdapter.io.axi <> modMem.io.axi
  modSim.reset := reset
  modSim.clock := sim_clock
  sim_wait := modSim.sim_wait
}
