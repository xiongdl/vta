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
import vta.util.config._
import vta.interface.axi._
import vta.interface.apb._
import vta.interface.ahb._
import vta.core._

/** Shell parameters. */
case class ShellParams(
    hostParams: AXIParams,
    memParams: AXIParams,
    vcrParams: VCRParams,
    vmeParams: VMEParams
)

case object ShellKey extends Field[ShellParams]

/** VTAShellInternal.
 *
 * Protocol-independent VTA core. Native host and memory bus front-ends are
 * selected by the external shell through VCRMaster and VMEMaster interfaces.
 */
class VTAShellInternal(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val vcr = Flipped(new VCRMaster)
    val vme = new VMEMaster
    val vmePerf = Input(new VMEPerfEvents)
  })

  val core = Module(new Core)

  core.io.vcr <> io.vcr
  core.io.vme <> io.vme
  core.io.vmePerf := io.vmePerf
}

/** VTAShell.
 *
 * AXI baseline external shell. Keeping this class and its ports unchanged
 * preserves existing RTL integrations and provides the reference behavior for
 * future APB host and AHB memory shell variants.
 */
class VTAShell(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val host = new AXILiteClient(p(ShellKey).hostParams)
    val mem = new AXIMaster(p(ShellKey).memParams)
  })

  val vcr = Module(new VCR)
  val vme = Module(new VME)
  val shell = Module(new VTAShellInternal)

  vcr.io.host <> io.host
  shell.io.vcr <> vcr.io.vcr
  shell.io.vme <> vme.io.vme
  vme.io.launch := vcr.io.vcr.launch
  shell.io.vmePerf := vme.io.perf
  io.mem <> vme.io.mem
}

/** Native APB4 host and AXI4 memory VTA shell. */
class VTAShellAPB(implicit p: Parameters) extends Module {
  private val hp = p(ShellKey).hostParams
  private val ap = APBParams(addrBits = hp.addrBits, dataBits = hp.dataBits)

  val io = IO(new Bundle {
    val host = new APBSlave(ap)
    val mem = new AXIMaster(p(ShellKey).memParams)
  })

  val vcr = Module(new VCRAPB)
  val vme = Module(new VME)
  val shell = Module(new VTAShellInternal)

  vcr.io.host <> io.host
  shell.io.vcr <> vcr.io.vcr
  shell.io.vme <> vme.io.vme
  vme.io.launch := vcr.io.vcr.launch
  shell.io.vmePerf := vme.io.perf
  io.mem <> vme.io.mem
}

/** Native AXI4-Lite host and AHB-Lite memory VTA shell. */
class VTAShellAHB(implicit p: Parameters) extends Module {
  private val mp = p(ShellKey).memParams
  private val ap = AHBParams(addrBits = mp.addrBits, dataBits = mp.dataBits)

  val io = IO(new Bundle {
    val host = new AXILiteClient(p(ShellKey).hostParams)
    val mem = new AHBMaster(ap)
  })

  val vcr = Module(new VCR)
  val vme = Module(new VMEAHB)
  val shell = Module(new VTAShellInternal)

  vcr.io.host <> io.host
  shell.io.vcr <> vcr.io.vcr
  shell.io.vme <> vme.io.vme
  vme.io.launch := vcr.io.vcr.launch
  shell.io.vmePerf := vme.io.perf
  io.mem <> vme.io.mem
}

/** Native APB4 host and AHB-Lite memory VTA shell. */
class VTAShellAPBAHB(implicit p: Parameters) extends Module {
  private val hp = p(ShellKey).hostParams
  private val mp = p(ShellKey).memParams
  private val hostAP = APBParams(addrBits = hp.addrBits, dataBits = hp.dataBits)
  private val memAP = AHBParams(addrBits = mp.addrBits, dataBits = mp.dataBits)

  val io = IO(new Bundle {
    val host = new APBSlave(hostAP)
    val mem = new AHBMaster(memAP)
  })

  val vcr = Module(new VCRAPB)
  val vme = Module(new VMEAHB)
  val shell = Module(new VTAShellInternal)

  vcr.io.host <> io.host
  shell.io.vcr <> vcr.io.vcr
  shell.io.vme <> vme.io.vme
  vme.io.launch := vcr.io.vcr.launch
  shell.io.vmePerf := vme.io.perf
  io.mem <> vme.io.mem
}
