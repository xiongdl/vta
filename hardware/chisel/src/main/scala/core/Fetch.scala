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

import chisel3._
import chisel3.util._
import vta.util.config._
import vta.shell._
import vta.util._

/** Fetch.
 *
 * Instructions are fetched into a shared single-port memory, classified into
 * narrow Load, Compute and Store address FIFOs, and prefetched into one next
 * register per execution stage.  The execution stages keep only their current
 * instruction register.
 */
class Fetch(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val vp = p(ShellKey).vcrParams
  val mp = p(ShellKey).memParams
  val io = IO(new Bundle {
    val launch = Input(Bool())
    val ins_baddr = Input(UInt(mp.addrBits.W))
    val ins_count = Input(UInt(vp.regBits.W))
    val vme_rd = new VMEReadMaster
    val inst = new Bundle {
      val ld = Decoupled(UInt(INST_BITS.W))
      val co = Decoupled(UInt(INST_BITS.W))
      val st = Decoupled(UInt(INST_BITS.W))
    }
  })

  if (mp.dataBits <= INST_BITS) {
    val fetch = Module(new FetchInstMemNarrow(debug))
    io <> fetch.io
  } else {
    val fetch = Module(new FetchInstMemWide(debug))
    io <> fetch.io
  }

}
