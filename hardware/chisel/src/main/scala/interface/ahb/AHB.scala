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

package vta.interface.ahb

import chisel3._
import chisel3.util._
import vta.util.genericbundle._

case class AHBParams(addrBits: Int = 32, dataBits: Int = 64) {
  require(addrBits > 0)
  require(dataBits >= 8 && isPow2(dataBits) && dataBits % 8 == 0)

  val sizeBits = 3
  val burstBits = 3
  val protBits = 4
  val transBits = 2
  val beatBytes = dataBits / 8
  val sizeConst = log2Ceil(beatBytes)
}

object AHBTransfer {
  val idle = 0.U(2.W)
  val busy = 1.U(2.W)
  val nonseq = 2.U(2.W)
  val seq = 3.U(2.W)
}

/** AHB-Lite master port. */
class AHBMaster(params: AHBParams) extends GenericParameterizedBundle(params) {
  val haddr = Output(UInt(params.addrBits.W))
  val hburst = Output(UInt(params.burstBits.W))
  val hprot = Output(UInt(params.protBits.W))
  val hsize = Output(UInt(params.sizeBits.W))
  val htrans = Output(UInt(params.transBits.W))
  val hwdata = Output(UInt(params.dataBits.W))
  val hwrite = Output(Bool())
  val hrdata = Input(UInt(params.dataBits.W))
  val hready = Input(Bool())
  val hresp = Input(Bool())
}

/** AHB-Lite slave port. */
class AHBSlave(params: AHBParams) extends GenericParameterizedBundle(params) {
  val haddr = Input(UInt(params.addrBits.W))
  val hburst = Input(UInt(params.burstBits.W))
  val hprot = Input(UInt(params.protBits.W))
  val hsize = Input(UInt(params.sizeBits.W))
  val htrans = Input(UInt(params.transBits.W))
  val hwdata = Input(UInt(params.dataBits.W))
  val hwrite = Input(Bool())
  val hrdata = Output(UInt(params.dataBits.W))
  val hready = Output(Bool())
  val hresp = Output(Bool())
}
