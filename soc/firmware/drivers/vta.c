#include "vta_soc.h"

static inline volatile uint32_t *vta_register(uint32_t offset) {
  return (volatile uint32_t *)(uintptr_t)(VTA_VCR_BASE + offset);
}

uint32_t vta_read(uint32_t offset) { return *vta_register(offset); }

void vta_write(uint32_t offset, uint32_t value) {
  *vta_register(offset) = value;
}

int vta_start(const struct vta_job *job) {
  if (job == 0 || job->insn_count == 0) return -1;
  if (vta_read(VTA_REG_CONTROL) & VTA_CONTROL_START) return -2;
  vta_write(VTA_REG_INSN_COUNT, job->insn_count);
  vta_write(VTA_REG_INSN_ADDR, job->insn_addr);
  vta_write(VTA_REG_CONTROL, VTA_CONTROL_START);
  return 0;
}

int vta_wait(uint32_t timeout_cycles, uint32_t *elapsed_cycles) {
  uint32_t start = vta_read(VTA_REG_CYCLE_COUNT);
  for (uint32_t polls = 0; polls < timeout_cycles; ++polls) {
    if (vta_read(VTA_REG_CONTROL) & VTA_CONTROL_DONE) {
      if (elapsed_cycles != 0)
        *elapsed_cycles = vta_read(VTA_REG_CYCLE_COUNT) - start;
      return 0;
    }
  }
  return -1;
}

