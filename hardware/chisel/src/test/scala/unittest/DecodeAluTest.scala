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

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import vta.core.{ComputeDecode, FetchDecode}

/** Keep every supported ALU opcode on the Fetch-to-Compute dispatch path. */
class DecodeAluTest extends AnyFlatSpec with ChiselScalatestTester {
  // Captured from a real Q31 requantize workload.  Task opcode 4 selects ALU
  // and ALU opcode 4 selects multiply.
  private val vmul = BigInt("0000c002000008000032000802401104", 16)

  behavior of "VTA ALU decode"

  it should "dispatch VMUL from Fetch to Compute" in {
    test(new FetchDecode) { dut =>
      dut.io.inst.poke(vmul.U)
      dut.io.isLoad.expect(false.B)
      dut.io.isStore.expect(false.B)
      dut.io.isCompute.expect(true.B)
    }

    test(new ComputeDecode) { dut =>
      dut.io.inst.poke(vmul.U)
      dut.io.isAlu.expect(true.B)
      dut.io.isGemm.expect(false.B)
      dut.io.isFinish.expect(false.B)
    }
  }
}
