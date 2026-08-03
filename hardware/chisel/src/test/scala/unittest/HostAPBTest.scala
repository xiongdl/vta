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

class HostAPBTester(c: VTAHostDPIToAPB) extends PeekPokeTester(c) {
  poke(c.io.dpi.req.valid, 0)
  poke(c.io.dpi.req.opcode, 0)
  poke(c.io.dpi.req.addr, 0)
  poke(c.io.dpi.req.value, 0)
  poke(c.io.apb.pready, 1)
  poke(c.io.apb.pslverr, 0)
  poke(c.io.apb.prdata, 0)
  step(1)

  // Read setup phase.
  poke(c.io.dpi.req.valid, 1)
  poke(c.io.dpi.req.opcode, 0)
  poke(c.io.dpi.req.addr, 0x24)
  expect(c.io.apb.psel, 1)
  expect(c.io.apb.penable, 0)
  expect(c.io.apb.pwrite, 0)
  expect(c.io.apb.paddr, 0x24)
  expect(c.io.dpi.req.deq, 1)
  step(1)

  // Read access and response phase.
  poke(c.io.dpi.req.valid, 0)
  poke(c.io.apb.prdata, 0x12345678)
  expect(c.io.apb.psel, 1)
  expect(c.io.apb.penable, 1)
  expect(c.io.dpi.resp.valid, 1)
  expect(c.io.dpi.resp.bits, 0x12345678)
  step(1)

  // Write setup phase.
  poke(c.io.dpi.req.valid, 1)
  poke(c.io.dpi.req.opcode, 1)
  poke(c.io.dpi.req.addr, 0x0c)
  poke(c.io.dpi.req.value, 0x76543210)
  expect(c.io.apb.psel, 1)
  expect(c.io.apb.penable, 0)
  expect(c.io.apb.pwrite, 1)
  expect(c.io.apb.paddr, 0x0c)
  expect(c.io.apb.pwdata, 0x76543210)
  expect(c.io.dpi.req.deq, 1)
  step(1)

  // Write access phase has no DPI read response.
  poke(c.io.dpi.req.valid, 0)
  expect(c.io.apb.psel, 1)
  expect(c.io.apb.penable, 1)
  expect(c.io.dpi.resp.valid, 0)
  step(1)
}

class HostAPBTest extends GenericTest(
  "HostAPBTest",
  (p: Parameters) => new VTAHostDPIToAPB()(p),
  (c: VTAHostDPIToAPB) => new HostAPBTester(c))
