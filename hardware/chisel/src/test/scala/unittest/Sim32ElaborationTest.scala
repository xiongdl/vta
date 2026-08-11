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
import org.scalatest.flatspec.AnyFlatSpec
import vta.TestSim32Config
import vta.test.Test
import vta.util.config.Parameters

/** Ensure the complete TSIM design elaborates with a 32-bit memory bus. */
class Sim32ElaborationTest extends AnyFlatSpec {
  behavior of "32-bit memory-bus TSIM"

  it should "elaborate the complete testbench" in {
    implicit val p: Parameters = new TestSim32Config
    (new ChiselStage).emitSystemVerilog(new Test,
      Array("--infer-rw", "--target-dir", "../../build/chisel",
        "-o", "Test.Sim32Elaboration"))
  }
}
