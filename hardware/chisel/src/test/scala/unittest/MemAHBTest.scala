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

import chiseltest.iotesters._
import vta.dpi._
import vta.util.config._

class MemAHBTester(c: VTAMemDPIToAHB) extends PeekPokeTester(c) {
  poke(c.io.ahb.haddr, 0)
  poke(c.io.ahb.hburst, 0)
  poke(c.io.ahb.hprot, 3)
  poke(c.io.ahb.hsize, 3)
  poke(c.io.ahb.htrans, 0)
  poke(c.io.ahb.hwdata, 0)
  poke(c.io.ahb.hwrite, 0)
  poke(c.io.dpi.rd.valid, 0)
  poke(c.io.dpi.rd.bits.data, 0)
  poke(c.io.dpi.rd.bits.id, 0)
  step(1)

  // Read address phase creates a single-beat DPI read request.
  poke(c.io.ahb.haddr, 0x1000)
  poke(c.io.ahb.htrans, 2)
  expect(c.io.ahb.hready, 1)
  expect(c.io.dpi.req.ar_valid, 1)
  expect(c.io.dpi.req.ar_addr, 0x1000)
  expect(c.io.dpi.req.ar_len, 0)
  step(1)

  // Hold the AHB data phase until the DPI read response arrives.
  poke(c.io.ahb.htrans, 0)
  expect(c.io.ahb.hready, 0)
  expect(c.io.dpi.rd.ready, 1)
  poke(c.io.dpi.rd.valid, 1)
  poke(c.io.dpi.rd.bits.data, BigInt("123456789abcdef0", 16))
  expect(c.io.ahb.hready, 1)
  expect(c.io.ahb.hrdata, BigInt("123456789abcdef0", 16))
  step(1)

  // Write address phase precedes its AHB/DPI data phase by one cycle.
  poke(c.io.dpi.rd.valid, 0)
  poke(c.io.ahb.haddr, 0x2000)
  poke(c.io.ahb.htrans, 2)
  poke(c.io.ahb.hwrite, 1)
  expect(c.io.ahb.hready, 1)
  expect(c.io.dpi.req.aw_valid, 1)
  expect(c.io.dpi.req.aw_addr, 0x2000)
  expect(c.io.dpi.req.aw_len, 0)
  step(1)

  poke(c.io.ahb.htrans, 0)
  poke(c.io.ahb.hwdata, BigInt("fedcba9876543210", 16))
  expect(c.io.ahb.hready, 1)
  expect(c.io.dpi.wr.valid, 1)
  expect(c.io.dpi.wr.bits.data, BigInt("fedcba9876543210", 16))
  expect(c.io.dpi.wr.bits.strb, 0xff)
  expect(c.io.ahb.hresp, 0)
  step(1)

  // A four-beat AHB burst creates one length-3 DPI request and then streams
  // four DPI responses across SEQ address phases.
  poke(c.io.ahb.haddr, 0x3000)
  poke(c.io.ahb.hburst, 3)
  poke(c.io.ahb.htrans, 2)
  poke(c.io.ahb.hwrite, 0)
  expect(c.io.dpi.req.ar_valid, 1)
  expect(c.io.dpi.req.ar_len, 3)
  step(1)

  for (beat <- 0 until 4) {
    poke(c.io.ahb.haddr, 0x3008 + beat * 8)
    poke(c.io.ahb.htrans, if (beat == 3) 0 else 3)
    poke(c.io.dpi.rd.valid, 1)
    poke(c.io.dpi.rd.bits.data, 0x40 + beat)
    expect(c.io.ahb.hready, 1)
    expect(c.io.ahb.hrdata, 0x40 + beat)
    step(1)
  }

  // An undefined-length INCR maps each AHB beat to a DPI SINGLE request.
  poke(c.io.dpi.rd.valid, 0)
  poke(c.io.ahb.haddr, 0x3800)
  poke(c.io.ahb.hburst, 1)
  poke(c.io.ahb.htrans, 2)
  expect(c.io.dpi.req.ar_valid, 1)
  expect(c.io.dpi.req.ar_len, 0)
  step(1)

  for (beat <- 0 until 3) {
    poke(c.io.ahb.haddr, 0x3808 + beat * 8)
    poke(c.io.ahb.htrans, if (beat == 2) 0 else 3)
    poke(c.io.dpi.rd.valid, 1)
    poke(c.io.dpi.rd.bits.data, 0x70 + beat)
    expect(c.io.ahb.hready, 1)
    expect(c.io.dpi.req.ar_valid, if (beat == 2) 0 else 1)
    if (beat != 2) {
      expect(c.io.dpi.req.ar_addr, 0x3808 + beat * 8)
      expect(c.io.dpi.req.ar_len, 0)
    }
    step(1)
  }

  // BUSY pauses an undefined write burst; the following SEQ issues the next
  // DPI SINGLE without consuming write data during the gap.
  poke(c.io.dpi.rd.valid, 0)
  poke(c.io.ahb.haddr, 0x3c00)
  poke(c.io.ahb.hburst, 1)
  poke(c.io.ahb.htrans, 2)
  poke(c.io.ahb.hwrite, 1)
  expect(c.io.dpi.req.aw_valid, 1)
  expect(c.io.dpi.req.aw_len, 0)
  step(1)

  poke(c.io.ahb.hwdata, 0x80)
  poke(c.io.ahb.htrans, 1)
  expect(c.io.dpi.wr.valid, 1)
  expect(c.io.dpi.req.aw_valid, 0)
  step(1)

  poke(c.io.ahb.haddr, 0x3c08)
  poke(c.io.ahb.htrans, 3)
  expect(c.io.dpi.wr.valid, 0)
  expect(c.io.dpi.req.aw_valid, 1)
  expect(c.io.dpi.req.aw_addr, 0x3c08)
  step(1)

  poke(c.io.ahb.hwdata, 0x81)
  poke(c.io.ahb.haddr, 0x3c10)
  poke(c.io.ahb.htrans, 3)
  expect(c.io.dpi.wr.valid, 1)
  expect(c.io.dpi.req.aw_valid, 1)
  expect(c.io.dpi.req.aw_addr, 0x3c10)
  step(1)

  poke(c.io.ahb.hwdata, 0x82)
  poke(c.io.ahb.htrans, 0)
  expect(c.io.dpi.wr.valid, 1)
  expect(c.io.dpi.req.aw_valid, 0)
  step(1)

  // BUSY can be inserted between write beats without consuming DPI data.
  poke(c.io.dpi.rd.valid, 0)
  poke(c.io.ahb.haddr, 0x4000)
  poke(c.io.ahb.hburst, 3)
  poke(c.io.ahb.htrans, 2)
  poke(c.io.ahb.hwrite, 1)
  expect(c.io.dpi.req.aw_valid, 1)
  expect(c.io.dpi.req.aw_len, 3)
  step(1)

  poke(c.io.ahb.hwdata, 0x50)
  poke(c.io.ahb.htrans, 1)
  expect(c.io.dpi.wr.valid, 1)
  expect(c.io.dpi.wr.bits.data, 0x50)
  step(1)

  // The BUSY address phase creates a gap in the write data stream.
  expect(c.io.dpi.wr.valid, 0)
  expect(c.io.ahb.hready, 1)
  step(1)

  poke(c.io.ahb.haddr, 0x4008)
  poke(c.io.ahb.htrans, 3)
  expect(c.io.dpi.wr.valid, 0)
  step(1)

  for (beat <- 1 until 4) {
    poke(c.io.ahb.hwdata, 0x50 + beat)
    poke(c.io.ahb.haddr, 0x4000 + (beat + 1) * 8)
    poke(c.io.ahb.htrans, if (beat == 3) 0 else 3)
    expect(c.io.dpi.wr.valid, 1)
    expect(c.io.dpi.wr.bits.data, 0x50 + beat)
    step(1)
  }
}

class MemAHBTest extends GenericTest(
  "MemAHBTest",
  (p: Parameters) => new VTAMemDPIToAHB()(p),
  (c: VTAMemDPIToAHB) => new MemAHBTester(c))
