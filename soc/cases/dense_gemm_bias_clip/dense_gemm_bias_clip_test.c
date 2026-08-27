#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sim_debug.h>

#include "dense_gemm_bias_clip_test.h"

#define NPU_BASE              0xf1000000U
#define VTA_CONTROL_REG       (NPU_BASE + 0x00U)
#define VTA_CYCLE_COUNT_REG   (NPU_BASE + 0x04U)
#define VTA_INSN_COUNT_REG    (NPU_BASE + 0x08U)
#define VTA_INSN_BASE_REG     (NPU_BASE + 0x0cU)
#define VTA_UOP_BASE_REG      (NPU_BASE + 0x10U)
#define VTA_INP_BASE_REG      (NPU_BASE + 0x14U)
#define VTA_WGT_BASE_REG      (NPU_BASE + 0x18U)
#define VTA_ACC_BASE_REG      (NPU_BASE + 0x1cU)
#define VTA_OUT_BASE_REG      (NPU_BASE + 0x20U)

#define MMIO32(address) (*(volatile uint32_t *)(uintptr_t)(address))
#define VTA_CONTROL_START 1U
#define VTA_STATUS_DONE   2U

int main(void) {
    uint32_t insn = (uint32_t)(uintptr_t)dense_gemm_bias_clip_insn;
    uint32_t uop = (uint32_t)(uintptr_t)dense_gemm_bias_clip_uop;
    uint32_t inp = (uint32_t)(uintptr_t)dense_gemm_bias_clip_inp;
    uint32_t wgt = (uint32_t)(uintptr_t)dense_gemm_bias_clip_wgt;
    uint32_t acc = (uint32_t)(uintptr_t)dense_gemm_bias_clip_acc;
    uint32_t out = (uint32_t)(uintptr_t)dense_gemm_bias_clip_out;

    printf("VTA case dense_gemm_bias_clip: insn=%u\r\n", (unsigned int)DENSE_GEMM_BIAS_CLIP_INSN_COUNT);
    printf("insn=%08x uop=%08x inp=%08x\r\n",
           (unsigned int)insn, (unsigned int)uop, (unsigned int)inp);
    printf("wgt=%08x acc=%08x out=%08x\r\n",
           (unsigned int)wgt, (unsigned int)acc, (unsigned int)out);

    MMIO32(VTA_INSN_COUNT_REG) = DENSE_GEMM_BIAS_CLIP_INSN_COUNT;
    MMIO32(VTA_INSN_BASE_REG) = insn;
    MMIO32(VTA_UOP_BASE_REG) = uop;
    MMIO32(VTA_INP_BASE_REG) = inp;
    MMIO32(VTA_WGT_BASE_REG) = wgt;
    MMIO32(VTA_ACC_BASE_REG) = acc;
    MMIO32(VTA_OUT_BASE_REG) = out;
    MMIO32(VTA_CONTROL_REG) = VTA_CONTROL_START;

    while (MMIO32(VTA_CONTROL_REG) != VTA_STATUS_DONE) {}
    uint32_t cycles = MMIO32(VTA_CYCLE_COUNT_REG);
    printf("VTA case counters: cycles=%u insns=%u\r\n",
           (unsigned int)cycles, (unsigned int)DENSE_GEMM_BIAS_CLIP_INSN_COUNT);

    for (size_t offset = 0; offset < DENSE_GEMM_BIAS_CLIP_EXPECTED_SIZE; offset += 4U) {
        uint32_t actual = *(volatile uint32_t *)((uintptr_t)dense_gemm_bias_clip_out + offset);
        uint32_t expected = *(const uint32_t *)((uintptr_t)dense_gemm_bias_clip_expected + offset);
        if (actual != expected) {
            printf("VTA case FAIL: offset=0x%zx expected=%08x actual=%08x\r\n",
                   offset, (unsigned int)expected, (unsigned int)actual);
            sim_finish();
            return 1;
        }
    }
    printf("VTA case PASS\r\n");
    sim_finish();
    return 0;
}
