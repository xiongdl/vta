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
import vta.core.TensorStoreWideVME
import vta.shell.ShellKey
import vta.util.config._

class TensorStoreWideVMEHandshakeTester(c: TensorStoreWideVME) extends PeekPokeTester(c) {
  // SOUT with ysize=1, xsize=32 and xstride=32.  A 128-bit memory beat holds
  // one output tensor, so this produces two commands of at most 16 beats.
  val inst = (BigInt(32) << 96) | (BigInt(32) << 80) | (BigInt(1) << 64) |
    (BigInt(4) << 7)

  poke(c.io.start, 0)
  poke(c.io.inst, inst)
  poke(c.io.baddr, 0x40000000L)
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
  val heldAddr = peek(c.io.vme_wr.cmd.bits.addr)
  val heldLen = peek(c.io.vme_wr.cmd.bits.len)
  val heldTag = peek(c.io.vme_wr.cmd.bits.tag)

  // A Decoupled producer must not advance its generator or change payload
  // while VALID is asserted and READY is withheld.
  for (_ <- 0 until 5) {
    expect(c.io.vme_wr.cmd.valid, 1)
    expect(c.io.vme_wr.cmd.bits.addr, heldAddr)
    expect(c.io.vme_wr.cmd.bits.len, heldLen)
    expect(c.io.vme_wr.cmd.bits.tag, heldTag)
    step(1)
  }

  poke(c.io.vme_wr.cmd.ready, 1)
  step(1)
  expect(c.io.vme_wr.cmd.valid, 0)
  expect(c.io.vme_wr.data.valid, 1)
}

class TensorStoreWideVMEHandshakeTest extends AnyFlatSpec with ChiselScalatestTester {
  private def wideStoreConfig: Parameters = {
    new Config((site, here, up) => {
      case ShellKey =>
        val shell = up(ShellKey)
        shell.copy(memParams = shell.memParams.copy(dataBits = 128))
    }) ++ new DefaultPynqConfig
  }

  behavior of "TensorStoreWideVME command handshakes"
  it should "hold command VALID and payload until READY" in {
    implicit val p: Parameters = wideStoreConfig
    test(new TensorStoreWideVME("out")).withAnnotations(Seq(TreadleBackendAnnotation))
      .runPeekPoke(new TensorStoreWideVMEHandshakeTester(_))
  }
}
