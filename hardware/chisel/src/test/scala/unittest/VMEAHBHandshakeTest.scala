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
import vta.TestDe10Config
import vta.shell._
import vta.util.config._

class VMEAHBHandshakeTester(c: VMEAHB) extends PeekPokeTester(c) {
  private val nReadClients = 5
  private val fullStrobe = 0xff

  poke(c.io.launch, 1)
  poke(c.io.mem.hrdata, 0)
  poke(c.io.mem.hready, 1)
  poke(c.io.mem.hresp, 0)
  for (i <- 0 until nReadClients) {
    poke(c.io.vme.rd(i).cmd.valid, 0)
    poke(c.io.vme.rd(i).cmd.bits.addr, 0)
    poke(c.io.vme.rd(i).cmd.bits.len, 0)
    poke(c.io.vme.rd(i).cmd.bits.tag, 0)
    poke(c.io.vme.rd(i).data.ready, 1)
  }
  poke(c.io.vme.wr(0).cmd.valid, 0)
  poke(c.io.vme.wr(0).cmd.bits.addr, 0)
  poke(c.io.vme.wr(0).cmd.bits.len, 0)
  poke(c.io.vme.wr(0).cmd.bits.tag, 0)
  poke(c.io.vme.wr(0).data.valid, 0)
  poke(c.io.vme.wr(0).data.bits.data, 0)
  poke(c.io.vme.wr(0).data.bits.strb, fullStrobe)
  step(1)

  // A waited read address phase must retain all address/control fields.
  poke(c.io.vme.rd(0).cmd.valid, 1)
  poke(c.io.vme.rd(0).cmd.bits.addr, 0x40010000L)
  poke(c.io.vme.rd(0).cmd.bits.len, 3)
  poke(c.io.vme.rd(0).cmd.bits.tag, 7)
  step(1)
  poke(c.io.vme.rd(0).cmd.valid, 0)
  poke(c.io.mem.hready, 0)
  for (_ <- 0 until 3) {
    expect(c.io.mem.htrans, 2)
    expect(c.io.mem.haddr, 0x40010000L)
    expect(c.io.mem.hburst, 3)
    expect(c.io.mem.hsize, 3)
    expect(c.io.mem.hwrite, 0)
    step(1)
  }
  poke(c.io.mem.hready, 1)
  step(1)

  // The first read data phase is extended while the following SEQ address
  // phase remains stable.  HRDATA is sampled only in the completing cycle.
  poke(c.io.mem.hready, 0)
  poke(c.io.mem.hrdata, 0xdead)
  for (_ <- 0 until 2) {
    expect(c.io.mem.htrans, 3)
    expect(c.io.mem.haddr, 0x40010008L)
    expect(c.io.mem.hburst, 3)
    expect(c.io.mem.hwrite, 0)
    step(1)
  }
  poke(c.io.mem.hrdata, 0x100)
  poke(c.io.mem.hready, 1)
  step(1)
  expect(c.io.vme.rd(0).data.valid, 1)
  expect(c.io.vme.rd(0).data.bits.data, 0x100)
  expect(c.io.vme.rd(0).data.bits.tag, 7)
  for (beat <- 1 until 4) {
    poke(c.io.mem.hrdata, 0x100 + beat)
    step(1)
  }
  step(1)

  // Queue the command and first write beat together.  The address phase and
  // the following write data phase must both tolerate arbitrary wait states.
  poke(c.io.vme.wr(0).cmd.valid, 1)
  poke(c.io.vme.wr(0).cmd.bits.addr, 0x40020000L)
  poke(c.io.vme.wr(0).cmd.bits.len, 3)
  poke(c.io.vme.wr(0).data.valid, 1)
  poke(c.io.vme.wr(0).data.bits.data, 0x200)
  step(1)
  poke(c.io.vme.wr(0).cmd.valid, 0)
  poke(c.io.vme.wr(0).data.valid, 0)
  step(1)

  poke(c.io.mem.hready, 0)
  for (_ <- 0 until 2) {
    expect(c.io.mem.htrans, 2)
    expect(c.io.mem.haddr, 0x40020000L)
    expect(c.io.mem.hburst, 3)
    expect(c.io.mem.hwrite, 1)
    step(1)
  }
  poke(c.io.mem.hready, 1)
  step(1)

  // With no second data beat available the manager inserts BUSY.  AMBA AHB
  // permits BUSY-to-SEQ during a waited fixed burst, but once SEQ is selected
  // it and the address/control signals must remain stable until HREADY rises.
  poke(c.io.mem.hready, 0)
  expect(c.io.mem.htrans, 1)
  expect(c.io.mem.haddr, 0x40020008L)
  expect(c.io.mem.hwdata, 0x200)
  step(1)
  poke(c.io.vme.wr(0).data.valid, 1)
  poke(c.io.vme.wr(0).data.bits.data, 0x201)
  step(1)
  poke(c.io.vme.wr(0).data.valid, 0)
  for (_ <- 0 until 2) {
    expect(c.io.mem.htrans, 3)
    expect(c.io.mem.haddr, 0x40020008L)
    expect(c.io.mem.hburst, 3)
    expect(c.io.mem.hsize, 3)
    expect(c.io.mem.hwrite, 1)
    expect(c.io.mem.hwdata, 0x200)
    step(1)
  }
  poke(c.io.mem.hready, 1)
  step(1)

  // Data 0x201 now belongs to the accepted address 0x40020008.  Supply the
  // remaining beats early enough to keep the final two SEQ phases continuous.
  expect(c.io.mem.hwdata, 0x201)
  poke(c.io.vme.wr(0).data.valid, 1)
  poke(c.io.vme.wr(0).data.bits.data, 0x202)
  step(1)
  poke(c.io.vme.wr(0).data.bits.data, 0x203)
  step(1)
  poke(c.io.vme.wr(0).data.valid, 0)
  expect(c.io.mem.hwdata, 0x202)
  expect(c.io.mem.htrans, 3)
  step(1)
  expect(c.io.mem.hwdata, 0x203)
  expect(c.io.mem.htrans, 0)
  step(1)
  expect(c.io.vme.wr(0).ack, 1)
}

class VMEAHBErrorTester(c: VMEAHB) extends PeekPokeTester(c) {
  poke(c.io.launch, 1)
  poke(c.io.mem.hrdata, 0)
  poke(c.io.mem.hready, 1)
  poke(c.io.mem.hresp, 0)
  for (i <- 0 until 5) {
    poke(c.io.vme.rd(i).cmd.valid, 0)
    poke(c.io.vme.rd(i).cmd.bits.addr, 0)
    poke(c.io.vme.rd(i).cmd.bits.len, 0)
    poke(c.io.vme.rd(i).cmd.bits.tag, 0)
    poke(c.io.vme.rd(i).data.ready, 1)
  }
  poke(c.io.vme.wr(0).cmd.valid, 0)
  poke(c.io.vme.wr(0).cmd.bits.addr, 0)
  poke(c.io.vme.wr(0).cmd.bits.len, 0)
  poke(c.io.vme.wr(0).cmd.bits.tag, 0)
  poke(c.io.vme.wr(0).data.valid, 0)
  poke(c.io.vme.wr(0).data.bits.data, 0)
  poke(c.io.vme.wr(0).data.bits.strb, 0xff)
  step(1)

  poke(c.io.vme.rd(0).cmd.valid, 1)
  poke(c.io.vme.rd(0).cmd.bits.addr, 0x40030000L)
  poke(c.io.vme.rd(0).cmd.bits.len, 0)
  step(1)
  poke(c.io.vme.rd(0).cmd.valid, 0)
  step(1)

  // First ERROR cycle: HRESP is high but HREADY is low, so the transfer has
  // not completed and no VME response may be emitted.
  poke(c.io.mem.hresp, 1)
  poke(c.io.mem.hready, 0)
  expect(c.io.vme.rd(0).data.valid, 0)
  step(1)
  expect(c.io.vme.rd(0).data.valid, 0)

  // Second ERROR cycle completes the AHB transfer.  VME has no error return
  // channel, so the implementation deliberately treats this as a fatal
  // protocol-visible hardware assertion instead of returning corrupt data.
  poke(c.io.mem.hready, 1)
  step(1)
}

class VMEAHBHandshakeTest extends AnyFlatSpec with ChiselScalatestTester {
  implicit val p: Parameters = new TestDe10Config

  behavior of "VME AHB64 handshakes"
  it should "hold address, control, and write data across wait states" in {
    test(new VMEAHB).withAnnotations(Seq(TreadleBackendAnnotation))
      .runPeekPoke(new VMEAHBHandshakeTester(_))
  }

  it should "recognize the second cycle of an ERROR response" in {
    test(new VMEAHB).withAnnotations(Seq(TreadleBackendAnnotation))
      .runPeekPoke(new VMEAHBErrorTester(_))
  }
}
