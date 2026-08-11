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
import vta.DefaultPynqConfig
import vta.core.Fetch
import vta.shell.ShellKey
import vta.util.config._

/** Ensure the wide-bus Fetch implementation elaborates with two and four
 * instructions in each external-memory beat.
 */
class FetchWideElaborationTest extends AnyFlatSpec {
  private val targetDir = "../../build/chisel"

  private def wideConfig(dataBits: Int): Parameters = {
    new Config((site, here, up) => {
      case ShellKey =>
        val shell = up(ShellKey)
        shell.copy(memParams = shell.memParams.copy(dataBits = dataBits))
    }) ++ new DefaultPynqConfig
  }

  behavior of "FetchInstMemWide"

  it should "elaborate for a 256-bit memory bus" in {
    implicit val p: Parameters = wideConfig(256)
    (new ChiselStage).emitSystemVerilog(new Fetch,
      Array("--infer-rw", "--target-dir", targetDir, "-o", "Fetch.256"))
  }

  it should "elaborate for a 512-bit memory bus" in {
    implicit val p: Parameters = wideConfig(512)
    (new ChiselStage).emitSystemVerilog(new Fetch,
      Array("--infer-rw", "--target-dir", targetDir, "-o", "Fetch.512"))
  }
}
