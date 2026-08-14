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

import chisel3.util._
import chiseltest.iotesters._
import scala.util.Random
import unittest.util._
import vta.core._
import vta.util.config._

object Alu_ref {
  /* alu_ref
   *
   * This is a software function used as a reference for the hardware
   */
  def alu(opcode: Int, a: Array[Int], b: Array[Int], width: Int,
    variant: Int = 0, rounding: Boolean = false) : Array[Int] = {
    val size = a.length
    val mask = Helper.getMask(log2Ceil(width))
    val res = Array.fill(size) {0}

    if (opcode == 0) {
      for (i <- 0 until size) { // min
        res(i) = if (a(i) < b(i)) a(i) else b(i)
      }
    } else if (opcode == 1) { // max
      for (i <- 0 until size) {
        res(i) = if (a(i) < b(i)) b(i) else a(i)
      }
    } else if (opcode == 2) { // add
      for (i <- 0 until size) {
        res(i) = a(i) + b(i)
      }
    } else if (opcode == 3) { // signed-direction shift
      for (i <- 0 until size) {
        if (b(i) < 0) {
          require(!rounding)
          res(i) = a(i) << ((-b(i)) & mask).toInt
        } else if (!rounding) {
          res(i) = a(i) >> (b(i) & mask).toInt
        } else {
          val exponent = (b(i) & mask).toInt
          if (exponent == 0) {
            res(i) = a(i)
          } else {
            val remainderMask = (1 << exponent) - 1
            val remainder = a(i) & remainderMask
            val divided = a(i) >> exponent
            val threshold = (remainderMask >> 1) + (if (divided < 0) 1 else 0)
            res(i) = divided + (if (remainder > threshold) 1 else 0)
          }
        }
      }
    } else if (opcode == 4) { // multiply
      for (i <- 0 until size) {
        val product = a(i).toLong * b(i).toLong
        if (variant == 1) {
          val nudge = if (rounding) 1L << 30 else 0L
          res(i) = ((product + nudge) >> 31).toInt
        } else {
          require(!rounding)
          res(i) = product.toInt
        }
      }
    } else { // default
      for (i <- 0 until size) {
        res(i) = 0
      }
    }
    res
  }
}

class AluVectorTester(c: AluVector, seed: Int = 47) extends PeekPokeTester(c) {
  val r = new Random(seed)

  def runCase(op: Int, variant: Int, rounding: Boolean, in_a: Array[Int], in_b: Array[Int]): Unit = {
    // generate data based on bits
    val bits = c.io.acc_a.tensorElemBits
    val mask = Helper.getMask(bits)
    val res = Alu_ref.alu(op, in_a, in_b, bits, variant, rounding)

    for (i <- 0 until c.blockOut) {
      poke(c.io.acc_a.data.bits(0)(i), in_a(i) & mask)
      poke(c.io.acc_b.data.bits(0)(i), in_b(i) & mask)
    }
    poke(c.io.opcode, op)
    poke(c.io.variant, variant)
    poke(c.io.rounding, if (rounding) 1 else 0)

    poke(c.io.acc_a.data.valid, 1)
    poke(c.io.acc_b.data.valid, 1)

    step(1)

    poke(c.io.acc_a.data.valid, 0)
    poke(c.io.acc_b.data.valid, 0)

    // wait for valid signal
    while (peek(c.io.acc_y.data.valid) == BigInt(0)) {
      step(1) // advance clock
    }
    if (peek(c.io.acc_y.data.valid) == BigInt(1)) {
      for (i <- 0 until c.blockOut) {
        expect(c.io.acc_y.data.bits(0)(i), res(i) & mask)
      }
    }
  }

  val bits = c.io.acc_a.tensorElemBits
  val dataGen = new RandomArray(c.blockOut, bits, r)
  for (op <- 0 until ALU_OP_NUM) {
    val in_b = if (op == 3) Array.fill(c.blockOut)(r.nextInt(9) - 4) else dataGen.any
    runCase(op, 0, false, dataGen.any, in_b)
  }

  runCase(4, 1, true, dataGen.any, dataGen.any)
  runCase(3, 0, true, dataGen.any,
    Array.fill(c.blockOut)(r.nextInt(8)))
}

class AluTest extends GenericTest("AluTest", (p:Parameters) =>
  new AluVector()(p), (c:AluVector) => new AluVectorTester(c, 48))
