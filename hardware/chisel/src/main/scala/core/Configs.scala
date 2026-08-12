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

package vta.core

import vta.util.config._

/** CoreConfig.
 *
 * This is one supported configuration for VTA. This file will
 * be eventually filled out with class configurations that can be
 * mixed/matched with Shell configurations for different backends.
 */
class CoreConfig extends Config((site, here, up) => {
  case CoreKey =>
    CoreParams(
      batch = 1,
      blockOut = 16,
      blockOutFactor = 1,
      blockIn = 16,
      inpBits = 8,
      wgtBits = 8,
      uopBits = 32,
      accBits = 32,
      outBits = 8,
      uopMemDepth = 2048,
      inpMemDepth = 2048,
      wgtMemDepth = 1024,
      accMemDepth = 2048,
      outMemDepth = 2048,
      instQueueEntries = 32
    )
})

private object TSimEnv {
  def int(name: String): Int =
    sys.env.getOrElse(name, throw new IllegalArgumentException(s"Missing $name")).toInt
}

/** Core configuration supplied by the TSIM build from vta_config.json. */
class TSimCoreConfig extends Config((site, here, up) => {
  case CoreKey =>
    CoreParams(
      batch = TSimEnv.int("VTA_TSIM_BATCH"),
      blockOut = TSimEnv.int("VTA_TSIM_BLOCK_OUT"),
      blockOutFactor = 1,
      blockIn = TSimEnv.int("VTA_TSIM_BLOCK_IN"),
      inpBits = TSimEnv.int("VTA_TSIM_INP_BITS"),
      wgtBits = TSimEnv.int("VTA_TSIM_WGT_BITS"),
      uopBits = 32,
      accBits = TSimEnv.int("VTA_TSIM_ACC_BITS"),
      outBits = TSimEnv.int("VTA_TSIM_OUT_BITS"),
      uopMemDepth = TSimEnv.int("VTA_TSIM_UOP_DEPTH"),
      inpMemDepth = TSimEnv.int("VTA_TSIM_INP_DEPTH"),
      wgtMemDepth = TSimEnv.int("VTA_TSIM_WGT_DEPTH"),
      accMemDepth = TSimEnv.int("VTA_TSIM_ACC_DEPTH"),
      outMemDepth = TSimEnv.int("VTA_TSIM_OUT_DEPTH"),
      instQueueEntries = 32
    )
})
