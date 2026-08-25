package vta.socgen

import chisel3.stage.ChiselStage
import vta.core.CoreConfig
import vta.shell._
import vta.interface.axi.AXIParams
import vta.util.config.{Config, Parameters}

/** Offline generator for the frozen APB32-host SoC variants.
  *
  * The LiteX AHB32 host option is an AHB-to-APB bridge in front of this native
  * APB port, so one frozen core package is shared by both system host choices.
  */
object SoCVariant extends App {
  val memoryProtocol = sys.env.getOrElse("VTA_SOC_MEMORY_PROTOCOL", "axi4")
  val memoryBits = sys.env.getOrElse("VTA_SOC_MEMORY_BITS", "64").toInt
  require(Set("axi4", "ahb-lite").contains(memoryProtocol))
  require(Set(32, 64, 128).contains(memoryBits))

  class SoCShellConfig extends Config((site, here, up) => {
    case ShellKey => ShellParams(
      hostParams = AXIParams(addrBits = 16, dataBits = 32, idBits = 13, lenBits = 4),
      memParams = AXIParams(
        addrBits = 32,
        dataBits = memoryBits,
        userBits = 5,
        // VMEAHB splits long commands into fixed bursts of at most 16 beats.
        lenBits = 8,
        coherent = true),
      vcrParams = VCRParams(),
      vmeParams = VMEParams())
  })

  implicit val p: Parameters = new Config(new CoreConfig ++ new SoCShellConfig)
  (new ChiselStage).emitSystemVerilog(
    if (memoryProtocol == "ahb-lite") new VTAShellAPBAHB else new VTAShellAPB,
    args ++ Array("--infer-rw"))
}
