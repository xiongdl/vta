package vta.socgen

import chisel3.stage.ChiselStage
import vta.DefaultSim32Config
import vta.shell.VTAShellAPB
import vta.util.config.Parameters

/** Offline generator for the SoC SIM APB32 + AXI32 frozen package. */
object APBHostSim32 extends App {
  implicit val p: Parameters = new DefaultSim32Config
  (new ChiselStage).emitSystemVerilog(
    new VTAShellAPB,
    args ++ Array("--infer-rw"))
}
