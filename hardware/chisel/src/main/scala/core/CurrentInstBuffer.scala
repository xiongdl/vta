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

/** Current instruction register for an execution stage.
 *
 * Fetch owns the corresponding next instruction register.  This register can
 * consume that next instruction in the same cycle in which the current
 * instruction completes, so no empty cycle is introduced between tasks.
 */
class CurrentInstBuffer extends Module {
  val io = IO(new Bundle {
    val enq = Flipped(Decoupled(UInt(INST_BITS.W)))
    val deq = Decoupled(UInt(INST_BITS.W))
  })

  val current = Reg(UInt(INST_BITS.W))
  val valid = RegInit(false.B)

  io.deq.bits := current
  io.deq.valid := valid
  io.enq.ready := !valid || io.deq.ready

  when(io.enq.fire) {
    current := io.enq.bits
    valid := true.B
  }.elsewhen(io.deq.fire) {
    valid := false.B
  }
}
