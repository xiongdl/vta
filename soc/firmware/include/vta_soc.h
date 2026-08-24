#pragma once

#include <stdint.h>

#ifndef VTA_VCR_BASE
#define VTA_VCR_BASE 0xf1000000u
#endif

enum {
  VTA_REG_CONTROL = 0x00,
  VTA_REG_CYCLE_COUNT = 0x04,
  VTA_REG_INSN_COUNT = 0x08,
  VTA_REG_INSN_ADDR = 0x0c,
  VTA_REG_UOP_ADDR = 0x10,
  VTA_REG_INP_ADDR = 0x14,
  VTA_REG_WGT_ADDR = 0x18,
  VTA_REG_ACC_ADDR = 0x1c,
  VTA_REG_OUT_ADDR = 0x20,
  VTA_REG_UOP_COUNT = 0x24,
};

enum {
  VTA_CONTROL_START = 1u,
  VTA_CONTROL_DONE = 2u,
};

struct vta_job {
  uint32_t insn_addr;
  uint32_t insn_count;
};

uint32_t vta_read(uint32_t offset);
void vta_write(uint32_t offset, uint32_t value);
int vta_start(const struct vta_job *job);
int vta_wait(uint32_t timeout_cycles, uint32_t *elapsed_cycles);

