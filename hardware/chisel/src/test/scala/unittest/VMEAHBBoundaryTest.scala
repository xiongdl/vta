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
import vta.DefaultDe10Config
import vta.shell._
import vta.util.config._

class VMEAHBBoundaryTester(c: VMEAHB) extends PeekPokeTester(c) {
  private val nReadClients = 5

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
  poke(c.io.vme.wr(0).data.bits.strb, 0xff)
  step(1)

  private def burstCode(beats: Int): Int = beats match {
    case 16 => 7
    case 8 => 5
    case 4 => 3
    case 1 => 0
  }

  private def runRead(startAddress: Int, chunks: Seq[Int]): Unit = {
    poke(c.io.vme.rd(0).cmd.valid, 1)
    poke(c.io.vme.rd(0).cmd.bits.addr, startAddress)
    poke(c.io.vme.rd(0).cmd.bits.len, chunks.sum - 1)
    poke(c.io.vme.rd(0).cmd.bits.tag, 1)
    expect(c.io.vme.rd(0).cmd.ready, 1)
    step(1)
    poke(c.io.vme.rd(0).cmd.valid, 0)

    var address = startAddress
    var value = 1
    for (chunk <- chunks) {
      expect(c.io.mem.htrans, 2)
      expect(c.io.mem.haddr, address)
      expect(c.io.mem.hburst, burstCode(chunk))
      step(1)

      for (beat <- 0 until chunk) {
        poke(c.io.mem.hrdata, value)
        expect(c.io.mem.htrans, if (beat == chunk - 1) 0 else 3)
        if (beat != chunk - 1) {
          expect(c.io.mem.haddr, address + 8)
        }
        step(1)
        address += 8
        value += 1
      }
    }

    // Drain the final queued response before issuing another command.
    step(1)
  }

  // 0x3a0 has 12 64-bit beats before 0x400.  A 16-beat request therefore
  // becomes INCR8 + INCR4, followed by INCR4 in the next 1KB region.
  runRead(0x3a0, Seq(8, 4, 4))

  // Only one beat remains before 0x400, so the first transfer must be SINGLE.
  runRead(0x3f8, Seq(1, 1, 1, 1))

  // An INCR16 ending at 0x3f8 stays entirely inside the 1KB region.
  runRead(0x380, Seq(16))
}

class VMEAHBBoundaryTest extends AnyFlatSpec with ChiselScalatestTester {
  implicit val p: Parameters = new DefaultDe10Config

  behavior of "VMEAHBBoundaryTest"
  it should "split fixed bursts at 1KB boundaries" in {
    test(new VMEAHB).withAnnotations(Seq(TreadleBackendAnnotation))
      .runPeekPoke(new VMEAHBBoundaryTester(_))
  }
}
