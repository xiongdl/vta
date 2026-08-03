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

/** Simulation-only native APB4 host. */
class VTAHostAPB(implicit p: Parameters) extends Module {
  private val hp = p(ShellKey).hostParams
  private val ap = APBParams(addrBits = hp.addrBits, dataBits = hp.dataBits)

  val io = IO(new Bundle {
    val apb = new APBMaster(ap)
  })
  val hostDPI = Module(new VTAHostDPI)
  val hostAPB = Module(new VTAHostDPIToAPB)
  hostDPI.io.reset := reset
  hostDPI.io.clock := clock
  hostAPB.io.dpi <> hostDPI.io.dpi
  io.apb <> hostAPB.io.apb
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

/** Simulation-only native AHB-Lite memory. */
class VTAMemAHB(implicit p: Parameters) extends Module {
  private val mp = p(ShellKey).memParams
  private val ap = AHBParams(addrBits = mp.addrBits, dataBits = mp.dataBits)

  val io = IO(new Bundle {
    val ahb = new AHBSlave(ap)
  })
  val memDPI = Module(new VTAMemDPI)
  val memAHB = Module(new VTAMemDPIToAHB)
  memDPI.io.reset := reset
  memDPI.io.clock := clock
  memDPI.io.dpi <> memAHB.io.dpi
  io.ahb <> memAHB.io.ahb
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

/** Native APB-host simulation shell. */
class SimShellAPB(implicit p: Parameters) extends Module {
  private val hp = p(ShellKey).hostParams
  private val ap = APBParams(addrBits = hp.addrBits, dataBits = hp.dataBits)

  val mem = IO(new AXIClient(p(ShellKey).memParams))
  val host = IO(new APBMaster(ap))
  val sim_clock = IO(Input(Clock()))
  val sim_wait = IO(Output(Bool()))

  val modSim = Module(new VTASim)
  val modHost = Module(new VTAHostAPB)
  val modMem = Module(new VTAMem)

  mem <> modMem.io.axi
  host <> modHost.io.apb
  modSim.reset := reset
  modSim.clock := sim_clock
  sim_wait := modSim.sim_wait
}

/** Native AHB-memory simulation shell. */
class SimShellAHB(implicit p: Parameters) extends Module {
  private val mp = p(ShellKey).memParams
  private val ap = AHBParams(addrBits = mp.addrBits, dataBits = mp.dataBits)

  val mem = IO(new AHBSlave(ap))
  val host = IO(new AXILiteMaster(p(ShellKey).hostParams))
  val sim_clock = IO(Input(Clock()))
  val sim_wait = IO(Output(Bool()))

  val modSim = Module(new VTASim)
  val modHost = Module(new VTAHost)
  val modMem = Module(new VTAMemAHB)

  host <> modHost.io.axi
  mem <> modMem.io.ahb
  modSim.reset := reset
  modSim.clock := sim_clock
  sim_wait := modSim.sim_wait
}

/** Native APB-host/AHB-memory simulation shell. */
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
  val modHost = Module(new VTAHostAPB)
  val modMem = Module(new VTAMemAHB)

  host <> modHost.io.apb
  mem <> modMem.io.ahb
  modSim.reset := reset
  modSim.clock := sim_clock
  sim_wait := modSim.sim_wait
}
