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

import chisel3.stage.ChiselStage
import chiseltest._
import chiseltest.iotesters._
import org.scalatest.flatspec.AnyFlatSpec
import vta.TestDe10Config
import vta.shell._
import vta.util.config._

class VMEHandshakeTester(c: VME) extends PeekPokeTester(c) {
  private val nReadClients = 5

  poke(c.io.launch, 1)
  poke(c.io.mem.aw.ready, 0)
  poke(c.io.mem.w.ready, 0)
  poke(c.io.mem.b.valid, 0)
  poke(c.io.mem.b.bits.resp, 0)
  poke(c.io.mem.b.bits.id, 0)
  poke(c.io.mem.ar.ready, 0)
  poke(c.io.mem.r.valid, 0)
  poke(c.io.mem.r.bits.data, 0)
  poke(c.io.mem.r.bits.resp, 0)
  poke(c.io.mem.r.bits.last, 0)
  poke(c.io.mem.r.bits.id, 0)
  for (i <- 0 until nReadClients) {
    poke(c.io.vme.rd(i).cmd.valid, 0)
    poke(c.io.vme.rd(i).cmd.bits.addr, 0)
    poke(c.io.vme.rd(i).cmd.bits.len, 0)
    poke(c.io.vme.rd(i).cmd.bits.tag, 0)
    poke(c.io.vme.rd(i).data.ready, if (i == 0) 0 else 1)
  }
  poke(c.io.vme.wr(0).cmd.valid, 0)
  poke(c.io.vme.wr(0).cmd.bits.addr, 0)
  poke(c.io.vme.wr(0).cmd.bits.len, 0)
  poke(c.io.vme.wr(0).cmd.bits.tag, 0)
  poke(c.io.vme.wr(0).data.valid, 0)
  poke(c.io.vme.wr(0).data.bits.data, 0)
  poke(c.io.vme.wr(0).data.bits.strb, 0xff)
  step(1)

  // The client command is accepted even though the AXI receiver withholds
  // ARREADY.  ARVALID and its payload must then remain asserted and stable.
  poke(c.io.vme.rd(0).cmd.valid, 1)
  poke(c.io.vme.rd(0).cmd.bits.addr, 0x40010000L)
  poke(c.io.vme.rd(0).cmd.bits.len, 3)
  poke(c.io.vme.rd(0).cmd.bits.tag, 7)
  step(1)
  poke(c.io.vme.rd(0).cmd.valid, 0)
  step(1)
  expect(c.io.mem.ar.valid, 1)
  expect(c.io.mem.ar.bits.addr, 0x40010000L)
  expect(c.io.mem.ar.bits.len, 3)
  val firstReadId = peek(c.io.mem.ar.bits.id)
  step(3)
  expect(c.io.mem.ar.valid, 1)
  expect(c.io.mem.ar.bits.addr, 0x40010000L)
  expect(c.io.mem.ar.bits.len, 3)
  expect(c.io.mem.ar.bits.id, firstReadId)

  poke(c.io.mem.ar.ready, 1)
  step(1)
  poke(c.io.mem.ar.ready, 0)
  expect(c.io.mem.ar.valid, 0)

  // Issue a second request from a different client, then return it before the
  // first request to cover legal cross-ID response reordering.
  poke(c.io.vme.rd(3).cmd.valid, 1)
  poke(c.io.vme.rd(3).cmd.bits.addr, 0x40020000L)
  poke(c.io.vme.rd(3).cmd.bits.len, 0)
  poke(c.io.vme.rd(3).cmd.bits.tag, 9)
  step(1)
  poke(c.io.vme.rd(3).cmd.valid, 0)
  step(1)
  expect(c.io.mem.ar.valid, 1)
  val secondReadId = peek(c.io.mem.ar.bits.id)
  assert(secondReadId != firstReadId)
  poke(c.io.mem.ar.ready, 1)
  step(1)
  poke(c.io.mem.ar.ready, 0)

  poke(c.io.vme.rd(3).data.ready, 0)
  poke(c.io.mem.r.valid, 1)
  poke(c.io.mem.r.bits.data, BigInt("0123456789abcdef", 16))
  poke(c.io.mem.r.bits.last, 1)
  poke(c.io.mem.r.bits.id, secondReadId)
  expect(c.io.mem.r.ready, 1)
  step(1)
  poke(c.io.mem.r.valid, 0)
  expect(c.io.vme.rd(3).data.valid, 1)
  expect(c.io.vme.rd(3).data.bits.data, BigInt("0123456789abcdef", 16))
  expect(c.io.vme.rd(3).data.bits.last, 1)
  expect(c.io.vme.rd(3).data.bits.tag, 9)
  poke(c.io.vme.rd(3).data.ready, 1)
  step(1)

  // Return the complete four-beat first burst into a stalled client queue.
  for (beat <- 0 until 4) {
    poke(c.io.mem.r.valid, 1)
    poke(c.io.mem.r.bits.data, 0x100 + beat)
    poke(c.io.mem.r.bits.last, if (beat == 3) 1 else 0)
    poke(c.io.mem.r.bits.id, firstReadId)
    expect(c.io.mem.r.ready, 1)
    step(1)
  }
  poke(c.io.mem.r.valid, 0)
  expect(c.io.vme.rd(0).data.valid, 1)
  expect(c.io.vme.rd(0).data.bits.data, 0x100)
  expect(c.io.vme.rd(0).data.bits.last, 0)
  expect(c.io.vme.rd(0).data.bits.tag, 7)
  step(2)
  expect(c.io.vme.rd(0).data.valid, 1)
  poke(c.io.vme.rd(0).data.ready, 1)
  step(4)
  expect(c.io.vme.rd(0).data.valid, 0)

  // Fill a complete response queue, then prove that the next response is
  // backpressured until the client drains one entry.
  poke(c.io.vme.rd(2).data.ready, 0)
  poke(c.io.vme.rd(2).cmd.valid, 1)
  poke(c.io.vme.rd(2).cmd.bits.addr, 0x40030000L)
  poke(c.io.vme.rd(2).cmd.bits.len, 15)
  poke(c.io.vme.rd(2).cmd.bits.tag, 11)
  step(1)
  poke(c.io.vme.rd(2).cmd.valid, 0)
  step(1)
  val fullQueueReadId = peek(c.io.mem.ar.bits.id)
  poke(c.io.mem.ar.ready, 1)
  step(1)
  poke(c.io.mem.ar.ready, 0)
  for (beat <- 0 until 16) {
    poke(c.io.mem.r.valid, 1)
    poke(c.io.mem.r.bits.data, 0x200 + beat)
    poke(c.io.mem.r.bits.last, if (beat == 15) 1 else 0)
    poke(c.io.mem.r.bits.id, fullQueueReadId)
    expect(c.io.mem.r.ready, 1)
    step(1)
  }
  poke(c.io.mem.r.valid, 0)

  poke(c.io.vme.rd(2).cmd.valid, 1)
  poke(c.io.vme.rd(2).cmd.bits.addr, 0x40040000L)
  poke(c.io.vme.rd(2).cmd.bits.len, 0)
  poke(c.io.vme.rd(2).cmd.bits.tag, 12)
  step(1)
  poke(c.io.vme.rd(2).cmd.valid, 0)
  step(1)
  val blockedReadId = peek(c.io.mem.ar.bits.id)
  poke(c.io.mem.ar.ready, 1)
  step(1)
  poke(c.io.mem.ar.ready, 0)
  poke(c.io.mem.r.valid, 1)
  poke(c.io.mem.r.bits.data, 0x300)
  poke(c.io.mem.r.bits.last, 1)
  poke(c.io.mem.r.bits.id, blockedReadId)
  expect(c.io.mem.r.ready, 0)
  step(2)
  expect(c.io.mem.r.ready, 0)
  poke(c.io.vme.rd(2).data.ready, 1)
  step(1)
  expect(c.io.mem.r.ready, 1)
  step(1)
  poke(c.io.mem.r.valid, 0)

  // The DPI memory model can produce the first read beat in the same cycle as
  // the AR handshake.  The freshly issued slot must already be routable.
  poke(c.io.vme.rd(4).data.ready, 0)
  poke(c.io.vme.rd(4).cmd.valid, 1)
  poke(c.io.vme.rd(4).cmd.bits.addr, 0x40050000L)
  poke(c.io.vme.rd(4).cmd.bits.len, 0)
  poke(c.io.vme.rd(4).cmd.bits.tag, 13)
  step(1)
  poke(c.io.vme.rd(4).cmd.valid, 0)
  step(1)
  val zeroLatencyReadId = peek(c.io.mem.ar.bits.id)
  poke(c.io.mem.ar.ready, 1)
  poke(c.io.mem.r.valid, 1)
  poke(c.io.mem.r.bits.data, 0x400)
  poke(c.io.mem.r.bits.last, 1)
  poke(c.io.mem.r.bits.id, zeroLatencyReadId)
  expect(c.io.mem.r.ready, 1)
  step(1)
  poke(c.io.mem.ar.ready, 0)
  poke(c.io.mem.r.valid, 0)
  expect(c.io.vme.rd(4).data.valid, 1)
  expect(c.io.vme.rd(4).data.bits.tag, 13)
}

class VMEHandshakeTest extends AnyFlatSpec with ChiselScalatestTester {
  implicit val p: Parameters = new TestDe10Config

  behavior of "VME AXI handshakes"
  it should "hold ARVALID and response VALID independently of READY" in {
    test(new VME).withAnnotations(Seq(TreadleBackendAnnotation))
      .runPeekPoke(new VMEHandshakeTester(_))
  }
}

class VMEParameterTest extends AnyFlatSpec {
  private def vmeConfig(clients: Int, depth: Int, idBits: Int): Parameters = {
    new Config((site, here, up) => {
      case ShellKey =>
        val shell = up(ShellKey)
        shell.copy(
          memParams = shell.memParams.copy(idBits = idBits),
          vmeParams = VMEParams(
            nReadClients = clients,
            clientBits = math.max(1, chisel3.util.log2Ceil(clients)),
            RequestQueueDepth = depth))
    }) ++ new TestDe10Config
  }

  behavior of "VME parameters"

  it should "elaborate non-default client counts and a single outstanding slot" in {
    for ((clients, depth) <- Seq((3, 1), (7, 5))) {
      implicit val p: Parameters = vmeConfig(clients, depth, idBits = 8)
      (new ChiselStage).emitChirrtl(new VME)
    }
  }

  it should "reject an outstanding depth larger than the AXI ID space" in {
    implicit val p: Parameters = vmeConfig(clients = 5, depth = 17, idBits = 4)
    assertThrows[IllegalArgumentException] {
      (new ChiselStage).emitChirrtl(new VME)
    }
  }
}
