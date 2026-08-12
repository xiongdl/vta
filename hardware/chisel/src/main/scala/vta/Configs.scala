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

package vta

import chisel3._
import vta.util.config._
import vta.shell._
import vta.core._
import vta.test._

/** VTA.
 *
 * This file contains all the configurations supported by VTA.
 * These configurations are built in a mix/match form based on core
 * and shell configurations.
 */
class DefaultPynqConfig extends Config(new CoreConfig ++ new PynqConfig)
class DefaultF1Config extends Config(new CoreConfig ++ new F1Config)
class DefaultDe10Config extends Config(new CoreConfig ++ new De10Config)
class DefaultSim32Config extends Config(new CoreConfig ++ new Sim32Config)
class DefaultTSimConfig extends Config(new TSimCoreConfig ++ new TSimConfig)

/** Enable detailed VME counters for simulation and performance analysis. */
class VMEPerfConfig extends Config((site, here, up) => {
  case ShellKey =>
    val shell = up(ShellKey)
    shell.copy(vcrParams = shell.vcrParams.copy(enableVMEPerfCounters = true))
})

class TestPynqConfig extends Config(new VMEPerfConfig ++ new DefaultPynqConfig)
class TestF1Config extends Config(new VMEPerfConfig ++ new DefaultF1Config)
class TestDe10Config extends Config(new VMEPerfConfig ++ new DefaultDe10Config)
class TestSim32Config extends Config(new VMEPerfConfig ++ new DefaultSim32Config)
class TestTSimConfig extends Config(new VMEPerfConfig ++ new DefaultTSimConfig)

private object ChiselOptions {
  def inferReadWrite(args: Array[String]): Array[String] =
    args ++ Array("--infer-rw")
}

object DefaultPynqConfig extends App {
  implicit val p: Parameters = new DefaultPynqConfig
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new XilinxShell,
    ChiselOptions.inferReadWrite(args))
}

object DefaultF1Config extends App {
  implicit val p: Parameters = new DefaultF1Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new XilinxShell,
    ChiselOptions.inferReadWrite(args))
}

object DefaultDe10Config extends App {
  implicit val p: Parameters = new DefaultDe10Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new IntelShell,
    ChiselOptions.inferReadWrite(args))
}

object TestDefaultPynqConfig extends App {
  implicit val p: Parameters = new TestPynqConfig
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new Test,
    ChiselOptions.inferReadWrite(args))
}

object TestDefaultF1Config extends App {
  implicit val p: Parameters = new TestF1Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new Test,
    ChiselOptions.inferReadWrite(args))
}

object TestDefaultDe10Config extends App {
  implicit val p: Parameters = new TestDe10Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new Test,
    ChiselOptions.inferReadWrite(args))
}

/** Generate the 32-bit memory-bus TSIM testbench. */
object DefaultSim32Config extends App {
  implicit val p: Parameters = new DefaultSim32Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new IntelShell,
    ChiselOptions.inferReadWrite(args))
}

object TestDefaultSim32Config extends App {
  implicit val p: Parameters = new TestSim32Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new Test,
    ChiselOptions.inferReadWrite(args))
}

/** Generate a TSIM model matching vta_config.json. */
object DefaultTSimConfig extends App {
  implicit val p: Parameters = new DefaultTSimConfig
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new IntelShell,
    ChiselOptions.inferReadWrite(args))
}

object TestDefaultTSimConfig extends App {
  implicit val p: Parameters = new TestTSimConfig
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new Test,
    ChiselOptions.inferReadWrite(args))
}

/** Generate a VTA shell with native APB4 host and AXI4 memory interfaces. */
object APBHostDefaultDe10Config extends App {
  implicit val p: Parameters = new DefaultDe10Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new VTAShellAPB,
    ChiselOptions.inferReadWrite(args))
}

/** Generate the native APB-host TSIM testbench. */
object TestAPBDefaultDe10Config extends App {
  implicit val p: Parameters = new TestDe10Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new TestAPB,
    ChiselOptions.inferReadWrite(args))
}

/** Generate a native AXI4-Lite-host/AHB-Lite-memory VTA shell. */
object AHBMemoryDefaultDe10Config extends App {
  implicit val p: Parameters = new DefaultDe10Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new VTAShellAHB,
    ChiselOptions.inferReadWrite(args))
}

/** Generate a native APB4-host/AHB-Lite-memory VTA shell. */
object APBAHBDefaultDe10Config extends App {
  implicit val p: Parameters = new DefaultDe10Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new VTAShellAPBAHB,
    ChiselOptions.inferReadWrite(args))
}

object TestAHBDefaultDe10Config extends App {
  implicit val p: Parameters = new TestDe10Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new TestAHB,
    ChiselOptions.inferReadWrite(args))
}

object TestAPBAHBDefaultDe10Config extends App {
  implicit val p: Parameters = new TestDe10Config
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new TestAPBAHB,
    ChiselOptions.inferReadWrite(args))
}
