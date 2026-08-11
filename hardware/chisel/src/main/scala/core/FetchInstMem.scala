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
import vta.shell._
import vta.util.config._

/** Shared instruction-memory backend used by narrow and wide Fetch frontends. */
class FetchInstMemCore(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val entries = p(CoreKey).instQueueEntries
  val slotBits = log2Ceil(entries)
  val countBits = log2Ceil(entries + 1)
  val nClients = 3
  val idLoad = 0
  val idCompute = 1
  val idStore = 2

  require(isPow2(entries) && entries >= 32 && entries <= 256,
    "-F- Instruction-memory depth must be a power of two from 32 through 256")

  val io = IO(new Bundle {
    val launch = Input(Bool())
    val reserve = Flipped(Decoupled(UInt(countBits.W)))
    val write = Flipped(Decoupled(UInt(INST_BITS.W)))
    val inst = new Bundle {
      val ld = Decoupled(UInt(INST_BITS.W))
      val co = Decoupled(UInt(INST_BITS.W))
      val st = Decoupled(UInt(INST_BITS.W))
    }
  })

  val localReset = reset.asBool || io.launch

  // InferReadWrite merges the mutually exclusive ports into one 128-bit
  // synchronous read/write port during RTL emission.
  val instMem = SyncReadMem(entries, UInt(INST_BITS.W))
  val addrQueues = Seq.fill(nClients) {
    withReset(localReset) { Module(new Queue(UInt(slotBits.W), entries)) }
  }
  val freeSlots = withReset(localReset) {
    Module(new Queue(UInt(slotBits.W), entries))
  }
  val requestQueue = withReset(localReset) {
    Module(new Queue(UInt(2.W), nClients))
  }

  val nextData = Reg(Vec(nClients, UInt(INST_BITS.W)))
  val nextValid = RegInit(VecInit(Seq.fill(nClients)(false.B)))
  val readPending = RegInit(VecInit(Seq.fill(nClients)(false.B)))
  val outputs = Seq(io.inst.ld, io.inst.co, io.inst.st)
  for (i <- 0 until nClients) {
    outputs(i).bits := nextData(i)
    outputs(i).valid := nextValid(i)
  }

  // Reserve a complete burst before its external-memory command is issued.
  val freeCount = RegInit(entries.U(countBits.W))
  io.reserve.ready := freeCount >= io.reserve.bits

  // Allocate virgin slots first, then recycle slots returned by local reads.
  val virginSlot = RegInit(0.U(countBits.W))
  val useVirginSlot = virginSlot < entries.U
  val allocatedSlot = Mux(useVirginSlot,
    virginSlot(slotBits - 1, 0), freeSlots.io.deq.bits)
  val allocationReady = useVirginSlot || freeSlots.io.deq.valid

  val writeDecode = Module(new FetchDecode)
  writeDecode.io.inst := io.write.bits
  val knownInstruction = writeDecode.io.isLoad || writeDecode.io.isCompute ||
    writeDecode.io.isStore
  val writeTargetReady = MuxCase(false.B, Seq(
    writeDecode.io.isLoad -> addrQueues(idLoad).io.enq.ready,
    writeDecode.io.isCompute -> addrQueues(idCompute).io.enq.ready,
    writeDecode.io.isStore -> addrQueues(idStore).io.enq.ready))
  io.write.ready := allocationReady && writeTargetReady
  val instructionWrite = io.write.fire

  when(io.write.valid) {
    assert(knownInstruction, "-F- FetchInstMem: unknown instruction type")
  }

  when(instructionWrite) {
    instMem.write(allocatedSlot, io.write.bits)
    when(useVirginSlot) {
      virginSlot := virginSlot + 1.U
    }
  }
  freeSlots.io.deq.ready := instructionWrite && !useVirginSlot

  for (i <- 0 until nClients) {
    addrQueues(i).io.enq.bits := allocatedSlot
    addrQueues(i).io.enq.valid := instructionWrite && (i match {
      case `idLoad` => writeDecode.io.isLoad
      case `idCompute` => writeDecode.io.isCompute
      case `idStore` => writeDecode.io.isStore
    })
  }

  // Empty next registers submit one request each.  Once admitted, requests are
  // serviced strictly in requestQueue order.
  val requestArb = Module(new Arbiter(UInt(2.W), nClients))
  for (i <- 0 until nClients) {
    requestArb.io.in(i).valid := !nextValid(i) && !readPending(i) &&
      addrQueues(i).io.deq.valid
    requestArb.io.in(i).bits := i.U
  }
  requestQueue.io.enq <> requestArb.io.out

  val rIdle :: rWait :: Nil = Enum(2)
  val readState = RegInit(rIdle)
  val readClient = Reg(UInt(2.W))
  val readSlot = Reg(UInt(slotBits.W))

  val selectedAddrValid = MuxLookup(requestQueue.io.deq.bits,
    false.B, Seq(
      idLoad.U -> addrQueues(idLoad).io.deq.valid,
      idCompute.U -> addrQueues(idCompute).io.deq.valid,
      idStore.U -> addrQueues(idStore).io.deq.valid))
  val selectedAddr = MuxLookup(requestQueue.io.deq.bits,
    0.U, Seq(
      idLoad.U -> addrQueues(idLoad).io.deq.bits,
      idCompute.U -> addrQueues(idCompute).io.deq.bits,
      idStore.U -> addrQueues(idStore).io.deq.bits))
  when(requestQueue.io.deq.valid) {
    assert(selectedAddrValid,
      "-F- FetchInstMem: queued read request has no instruction address")
  }

  // Issue directly from the head request when the single memory port is not
  // being written.  Writes retain priority without an extra issue state.
  val localReadEnable = readState === rIdle && requestQueue.io.deq.valid &&
    !instructionWrite
  requestQueue.io.deq.ready := localReadEnable
  for (i <- 0 until nClients) {
    addrQueues(i).io.deq.ready := localReadEnable &&
      requestQueue.io.deq.bits === i.U
  }

  val localReadData = instMem.read(selectedAddr, localReadEnable)
  val completedLocalRead = readState === rWait

  when(io.launch) {
    readState := rIdle
  }.otherwise {
    switch(readState) {
      is(rIdle) {
        when(localReadEnable) {
          readClient := requestQueue.io.deq.bits
          readSlot := selectedAddr
          readState := rWait
        }
      }
      is(rWait) {
        readState := rIdle
      }
    }
  }

  // Once copied into next, the shared-memory slot can be reused.
  freeSlots.io.enq.valid := completedLocalRead
  freeSlots.io.enq.bits := readSlot
  assert(!completedLocalRead || freeSlots.io.enq.ready,
    "-F- FetchInstMem: free-slot FIFO overflow")

  val reserveSlots = io.reserve.fire
  val releaseSlot = completedLocalRead
  when(io.launch) {
    freeCount := entries.U
    virginSlot := 0.U
  }.elsewhen(reserveSlots && releaseSlot) {
    freeCount := freeCount - io.reserve.bits + 1.U
  }.elsewhen(reserveSlots) {
    freeCount := freeCount - io.reserve.bits
  }.elsewhen(releaseSlot) {
    freeCount := freeCount + 1.U
  }

  for (i <- 0 until nClients) {
    val nextPop = outputs(i).fire
    val nextPush = completedLocalRead && readClient === i.U
    when(io.launch) {
      nextValid(i) := false.B
      readPending(i) := false.B
    }.otherwise {
      when(nextPush) {
        nextData(i) := localReadData
        nextValid(i) := true.B
        readPending(i) := false.B
      }.elsewhen(nextPop) {
        nextValid(i) := false.B
      }
      when(requestArb.io.in(i).fire) {
        readPending(i) := true.B
      }
    }
  }

  if (debug) {
    when(instructionWrite) {
      printf("[Fetch] write slot=%x inst=%x\n", allocatedSlot, io.write.bits)
    }
    when(completedLocalRead) {
      printf("[Fetch] next client=%x slot=%x inst=%x\n",
        readClient, readSlot, localReadData)
    }
  }
}

/** Fetch frontend for a memory bus no wider than one instruction. */
class FetchInstMemNarrow(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val vp = p(ShellKey).vcrParams
  val mp = p(ShellKey).memParams
  val entries = p(CoreKey).instQueueEntries
  val countBits = log2Ceil(entries + 1)
  val beatsPerInst = INST_BITS / mp.dataBits
  val beatBits = math.max(1, log2Ceil(beatsPerInst))
  val bytesPerBeat = mp.dataBits / 8
  val maxBurstInsts = entries / 4
  val maxBurstBeats = maxBurstInsts * beatsPerInst
  val maxBurstBytes = maxBurstBeats * bytesPerBeat
  val burstCountBits = mp.lenBits + 1

  require(mp.dataBits <= INST_BITS)
  require(INST_BITS % mp.dataBits == 0)
  require(isPow2(beatsPerInst))
  require(maxBurstBeats <= (1 << mp.lenBits),
    "-F- Memory Bus length cannot carry one-quarter of the instruction memory")
  require(maxBurstBytes <= 1024 && 1024 % maxBurstBytes == 0,
    "-F- Maximum Fetch burst must divide the conservative 1KB bus boundary")

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

  val launchLast = RegNext(io.launch, init = false.B)
  val launch = io.launch && !launchLast
  val core = Module(new FetchInstMemCore(debug))
  core.io.launch := launch
  io.inst <> core.io.inst
  val dIdle :: dReserve :: dCommand :: dReceive :: Nil = Enum(4)
  val state = RegInit(dIdle)
  val address = Reg(UInt(mp.addrBits.W))
  val remaining = Reg(UInt(vp.regBits.W))
  val selectedInsts = Reg(UInt(countBits.W))
  val selectedBeats = Reg(UInt(burstCountBits.W))
  val beatsLeft = Reg(UInt(burstCountBits.W))
  val instBeat = RegInit(0.U(beatBits.W))
  val parts = Reg(Vec(beatsPerInst, UInt(mp.dataBits.W)))

  val wantedInsts = Mux(remaining > maxBurstInsts.U,
    maxBurstInsts.U, remaining(countBits - 1, 0))
  core.io.reserve.valid := state === dReserve
  core.io.reserve.bits := wantedInsts

  io.vme_rd.cmd.valid := state === dCommand
  io.vme_rd.cmd.bits.addr := address
  io.vme_rd.cmd.bits.len := (selectedBeats - 1.U)(mp.lenBits - 1, 0)
  io.vme_rd.cmd.bits.tag := 0.U

  val completedParts = Wire(Vec(beatsPerInst, UInt(mp.dataBits.W)))
  completedParts := parts
  completedParts(instBeat) := io.vme_rd.data.bits.data
  val finalInstBeat = instBeat === (beatsPerInst - 1).U
  core.io.write.valid := state === dReceive && io.vme_rd.data.valid && finalInstBeat
  core.io.write.bits := completedParts.asUInt
  io.vme_rd.data.ready := state === dReceive &&
    (!finalInstBeat || core.io.write.ready)
  val dataFire = io.vme_rd.data.fire

  when(dataFire) {
    parts(instBeat) := io.vme_rd.data.bits.data
    when(finalInstBeat) {
      instBeat := 0.U
    }.otherwise {
      instBeat := instBeat + 1.U
    }
    beatsLeft := beatsLeft - 1.U
    assert(io.vme_rd.data.bits.last === (beatsLeft === 1.U),
      "-F- FetchInstMemNarrow: malformed VME burst")
  }

  when(launch) {
    address := io.ins_baddr
    remaining := io.ins_count
    instBeat := 0.U
    state := Mux(io.ins_count === 0.U, dIdle, dReserve)
    assert((io.ins_baddr & (maxBurstBytes - 1).U) === 0.U,
      "-F- FetchInstMemNarrow: instruction address must align to maximum burst bytes")
  }.otherwise {
    switch(state) {
      is(dIdle) {
        // Wait for launch.
      }
      is(dReserve) {
        when(core.io.reserve.fire) {
          selectedInsts := wantedInsts
          selectedBeats := wantedInsts * beatsPerInst.U
          state := dCommand
        }
      }
      is(dCommand) {
        when(io.vme_rd.cmd.fire) {
          remaining := remaining - selectedInsts
          address := address + selectedBeats * (mp.dataBits / 8).U
          beatsLeft := selectedBeats
          state := dReceive
        }
      }
      is(dReceive) {
        when(dataFire && beatsLeft === 1.U) {
          state := Mux(remaining === 0.U, dIdle, dReserve)
        }
      }
    }
  }
}

/** Fetch frontend for a memory beat containing multiple instructions. */
class FetchInstMemWide(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val vp = p(ShellKey).vcrParams
  val mp = p(ShellKey).memParams
  val entries = p(CoreKey).instQueueEntries
  val countBits = log2Ceil(entries + 1)
  val instsPerBeat = mp.dataBits / INST_BITS
  val instIndexBits = math.max(1, log2Ceil(instsPerBeat))
  val bytesPerBeat = mp.dataBits / 8
  val maxBurstInsts = entries / 4
  val maxBurstBeats = maxBurstInsts / instsPerBeat
  val maxBurstBytes = maxBurstBeats * bytesPerBeat
  val burstCountBits = mp.lenBits + 1

  require(mp.dataBits > INST_BITS)
  require(mp.dataBits % INST_BITS == 0)
  require(isPow2(instsPerBeat))
  require(maxBurstInsts % instsPerBeat == 0,
    "-F- One-quarter of instruction memory must contain complete wide bus beats")
  require(maxBurstBeats > 0 && maxBurstBeats <= (1 << mp.lenBits),
    "-F- Memory Bus length cannot carry one-quarter of the instruction memory")
  require(maxBurstBytes <= 1024 && 1024 % maxBurstBytes == 0,
    "-F- Maximum Fetch burst must divide the conservative 1KB bus boundary")

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

  val launchLast = RegNext(io.launch, init = false.B)
  val launch = io.launch && !launchLast
  val core = Module(new FetchInstMemCore(debug))
  core.io.launch := launch
  io.inst <> core.io.inst
  val dIdle :: dReserve :: dCommand :: dReceive :: Nil = Enum(4)
  val state = RegInit(dIdle)
  val address = Reg(UInt(mp.addrBits.W))
  val remaining = Reg(UInt(vp.regBits.W))
  val selectedInsts = Reg(UInt(countBits.W))
  val selectedBeats = Reg(UInt(burstCountBits.W))
  val beatsLeft = Reg(UInt(burstCountBits.W))
  val burstInstsLeft = Reg(UInt(countBits.W))

  val beatData = Reg(UInt(mp.dataBits.W))
  val beatValid = RegInit(false.B)
  val beatInstIndex = RegInit(0.U(instIndexBits.W))
  val beatInstsLeft = Reg(UInt((instIndexBits + 1).W))
  val beatInstVec = beatData.asTypeOf(Vec(instsPerBeat, UInt(INST_BITS.W)))

  val wantedInsts = Mux(remaining > maxBurstInsts.U,
    maxBurstInsts.U, remaining(countBits - 1, 0))
  val wantedBeats = (wantedInsts + (instsPerBeat - 1).U) / instsPerBeat.U
  core.io.reserve.valid := state === dReserve
  core.io.reserve.bits := wantedInsts

  io.vme_rd.cmd.valid := state === dCommand
  io.vme_rd.cmd.bits.addr := address
  io.vme_rd.cmd.bits.len := (selectedBeats - 1.U)(mp.lenBits - 1, 0)
  io.vme_rd.cmd.bits.tag := 0.U

  io.vme_rd.data.ready := state === dReceive && !beatValid
  val dataFire = io.vme_rd.data.fire
  val instsInNewBeat = Mux(burstInstsLeft > instsPerBeat.U,
    instsPerBeat.U, burstInstsLeft)

  core.io.write.valid := beatValid
  core.io.write.bits := beatInstVec(beatInstIndex)
  val instructionFire = core.io.write.fire
  val finishingBeat = instructionFire && beatInstsLeft === 1.U
  val finishingBurst = finishingBeat && beatsLeft === 0.U

  when(dataFire) {
    beatData := io.vme_rd.data.bits.data
    beatValid := true.B
    beatInstIndex := 0.U
    beatInstsLeft := instsInNewBeat
    burstInstsLeft := burstInstsLeft - instsInNewBeat
    beatsLeft := beatsLeft - 1.U
    assert(io.vme_rd.data.bits.last === (beatsLeft === 1.U),
      "-F- FetchInstMemWide: malformed VME burst")
  }
  when(instructionFire) {
    when(beatInstsLeft === 1.U) {
      beatValid := false.B
    }.otherwise {
      beatInstIndex := beatInstIndex + 1.U
      beatInstsLeft := beatInstsLeft - 1.U
    }
  }

  when(launch) {
    address := io.ins_baddr
    remaining := io.ins_count
    beatValid := false.B
    state := Mux(io.ins_count === 0.U, dIdle, dReserve)
    assert((io.ins_baddr & (maxBurstBytes - 1).U) === 0.U,
      "-F- FetchInstMemWide: instruction address must align to maximum burst bytes")
  }.otherwise {
    switch(state) {
      is(dIdle) {
        // Wait for launch.
      }
      is(dReserve) {
        when(core.io.reserve.fire) {
          selectedInsts := wantedInsts
          selectedBeats := wantedBeats
          state := dCommand
        }
      }
      is(dCommand) {
        when(io.vme_rd.cmd.fire) {
          remaining := remaining - selectedInsts
          address := address + selectedBeats * (mp.dataBits / 8).U
          beatsLeft := selectedBeats
          burstInstsLeft := selectedInsts
          state := dReceive
        }
      }
      is(dReceive) {
        when(finishingBurst) {
          state := Mux(remaining === 0.U, dIdle, dReserve)
        }
      }
    }
  }
}
