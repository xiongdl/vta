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

package unittest

import chiseltest._
import chiseltest.iotesters._
import org.scalatest.flatspec.AnyFlatSpec
import vta.DefaultPynqConfig
import vta.core._
import vta.shell.ShellKey
import vta.util.config._

private object BaddrAdditionFixtures {
  val Baddr = BigInt("40000100", 16)
  val Expected = BigInt("40000200", 16)

  def memInst(op: Int, id: Int, dramOffset: Int, xsize: Int = 1): BigInt =
    BigInt(op) | (BigInt(id) << 7) | (BigInt(dramOffset) << 26) |
      (BigInt(1) << 64) | (BigInt(xsize) << 80) | (BigInt(xsize) << 96)

  def memoryConfig(dataBits: Int): Parameters = {
    new Config((site, here, up) => {
      case ShellKey =>
        val shell = up(ShellKey)
        shell.copy(memParams = shell.memParams.copy(dataBits = dataBits))
    }) ++ new DefaultPynqConfig
  }
}

class GenVMECmdWideBaddrTester(c: GenVMECmdWide) extends PeekPokeTester(c) {
  import BaddrAdditionFixtures._

  poke(c.io.start, 0)
  poke(c.io.isBusy, 0)
  poke(c.io.updateState, 0)
  poke(c.io.canSendCmd, 1)
  poke(c.io.baddr, Baddr)
  poke(c.io.ysize, 1)
  poke(c.io.xsize, 1)
  poke(c.io.xstride, 1)
  poke(c.io.dram_offset, 16) // 16 input tensors * 16 bytes = 0x100.
  poke(c.io.sram_offset, 0)
  poke(c.io.xpad_0, 0)
  poke(c.io.xpad_1, 0)
  poke(c.io.ypad_0, 0)
  poke(c.io.vmeCmd.ready, 0)
  step(1)

  poke(c.io.start, 1)
  poke(c.io.isBusy, 1)
  step(1)
  poke(c.io.start, 0)
  expect(c.io.vmeCmd.valid, 1)
  expect(c.io.vmeCmd.bits.addr, Expected)
}

class GenVMECmdNarrowBaddrTester(c: GenVMECmd) extends PeekPokeTester(c) {
  import BaddrAdditionFixtures._

  poke(c.io.start, 0)
  poke(c.io.isBusy, 0)
  poke(c.io.inst, memInst(op = 0, id = 2, dramOffset = 16))
  poke(c.io.baddr, Baddr)
  poke(c.io.vmeCmd.ready, 0)
  step(1)

  poke(c.io.start, 1)
  step(1)
  poke(c.io.start, 0)
  poke(c.io.isBusy, 1)
  step(1)
  expect(c.io.vmeCmd.valid, 1)
  expect(c.io.vmeCmd.bits.addr, Expected)
}

class TensorStoreNarrowBaddrTester(c: TensorStoreNarrowVME) extends PeekPokeTester(c) {
  import BaddrAdditionFixtures._

  poke(c.io.start, 0)
  poke(c.io.inst, memInst(op = 1, id = 4, dramOffset = 16))
  poke(c.io.baddr, Baddr)
  poke(c.io.vme_wr.cmd.ready, 0)
  poke(c.io.vme_wr.data.ready, 0)
  poke(c.io.vme_wr.ack, 0)
  poke(c.io.tensor.wr(0).valid, 0)
  poke(c.io.tensor.wr(0).bits.idx, 0)
  step(1)

  poke(c.io.start, 1)
  step(1)
  poke(c.io.start, 0)
  expect(c.io.vme_wr.cmd.valid, 1)
  expect(c.io.vme_wr.cmd.bits.addr, Expected)
}

class LoadUopSimpleBaddrTester(c: LoadUopSimple) extends PeekPokeTester(c) {
  import BaddrAdditionFixtures._

  poke(c.io.start, 0)
  poke(c.io.dec.op, 0)
  poke(c.io.dec.pop_prev, 0)
  poke(c.io.dec.pop_next, 0)
  poke(c.io.dec.push_prev, 0)
  poke(c.io.dec.push_next, 0)
  poke(c.io.dec.id, 0)
  poke(c.io.dec.sram_offset, 0)
  poke(c.io.dec.dram_offset, 64) // 64 uops * 4 bytes = 0x100.
  poke(c.io.dec.empty_0, 0)
  poke(c.io.dec.ysize, 1)
  poke(c.io.dec.xsize, 1)
  poke(c.io.dec.xstride, 1)
  poke(c.io.dec.ypad_0, 0)
  poke(c.io.dec.ypad_1, 0)
  poke(c.io.dec.xpad_0, 0)
  poke(c.io.dec.xpad_1, 0)
  poke(c.io.baddr, Baddr)
  poke(c.io.vme_rd.cmd.ready, 0)
  poke(c.io.vme_rd.data.valid, 0)
  poke(c.io.vme_rd.data.bits.data, 0)
  poke(c.io.vme_rd.data.bits.tag, 0)
  poke(c.io.vme_rd.data.bits.last, 0)
  poke(c.io.uop.idx.valid, 0)
  poke(c.io.uop.idx.bits, 0)
  step(1)

  poke(c.io.start, 1)
  step(1)
  poke(c.io.start, 0)
  expect(c.io.vme_rd.cmd.valid, 1)
  expect(c.io.vme_rd.cmd.bits.addr, Expected)
}

class TensorDataCtrlBaddrTester(c: TensorDataCtrl) extends PeekPokeTester(c) {
  import BaddrAdditionFixtures._

  poke(c.io.start, 0)
  poke(c.io.inst, memInst(op = 0, id = 2, dramOffset = 16))
  poke(c.io.baddr, Baddr)
  poke(c.io.xinit, 0)
  poke(c.io.xupdate, 0)
  poke(c.io.yupdate, 0)
  step(1)

  poke(c.io.start, 1)
  step(1)
  poke(c.io.start, 0)
  expect(c.io.addr, Expected)
}

class BaddrAdditionTest extends AnyFlatSpec with ChiselScalatestTester {
  import BaddrAdditionFixtures._

  behavior of "VTA DRAM base-address addition"

  it should "carry overlapping address bits in the wide load generator" in {
    implicit val p: Parameters = memoryConfig(128)
    test(new GenVMECmdWide("inp")).withAnnotations(Seq(TreadleBackendAnnotation))
      .runPeekPoke(new GenVMECmdWideBaddrTester(_))
  }

  it should "carry overlapping address bits in the narrow load generator" in {
    implicit val p: Parameters = memoryConfig(64)
    test(new GenVMECmd("inp")).withAnnotations(Seq(TreadleBackendAnnotation))
      .runPeekPoke(new GenVMECmdNarrowBaddrTester(_))
  }

  it should "carry overlapping address bits in the narrow store" in {
    implicit val p: Parameters = memoryConfig(64)
    test(new TensorStoreNarrowVME("out")).withAnnotations(Seq(TreadleBackendAnnotation))
      .runPeekPoke(new TensorStoreNarrowBaddrTester(_))
  }

  it should "carry overlapping address bits in the simple uop loader" in {
    implicit val p: Parameters = memoryConfig(64)
    test(new LoadUopSimple).withAnnotations(Seq(TreadleBackendAnnotation))
      .runPeekPoke(new LoadUopSimpleBaddrTester(_))
  }

  it should "carry overlapping address bits in the legacy tensor data controller" in {
    implicit val p: Parameters = memoryConfig(64)
    test(new TensorDataCtrl("inp")).withAnnotations(Seq(TreadleBackendAnnotation))
      .runPeekPoke(new TensorDataCtrlBaddrTester(_))
  }
}
