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

package vta.interface.apb

import chisel3._
import chisel3.util._
import vta.util.genericbundle._

/** APB4 interface parameters. */
case class APBParams(addrBits: Int = 32, dataBits: Int = 32) {
  require(addrBits > 0)
  require(dataBits >= 8 && isPow2(dataBits) && dataBits % 8 == 0)

  val strbBits = dataBits / 8
  val protBits = 3
}

/** APB4 slave port, with directions as seen by the slave. */
class APBSlave(params: APBParams) extends GenericParameterizedBundle(params) {
  val paddr = Input(UInt(params.addrBits.W))
  val psel = Input(Bool())
  val penable = Input(Bool())
  val pwrite = Input(Bool())
  val pwdata = Input(UInt(params.dataBits.W))
  val pstrb = Input(UInt(params.strbBits.W))
  val pprot = Input(UInt(params.protBits.W))
  val prdata = Output(UInt(params.dataBits.W))
  val pready = Output(Bool())
  val pslverr = Output(Bool())
}

/** APB4 master port, with directions as seen by the master. */
class APBMaster(params: APBParams) extends GenericParameterizedBundle(params) {
  val paddr = Output(UInt(params.addrBits.W))
  val psel = Output(Bool())
  val penable = Output(Bool())
  val pwrite = Output(Bool())
  val pwdata = Output(UInt(params.dataBits.W))
  val pstrb = Output(UInt(params.strbBits.W))
  val pprot = Output(UInt(params.protBits.W))
  val prdata = Input(UInt(params.dataBits.W))
  val pready = Input(Bool())
  val pslverr = Input(Bool())
}
