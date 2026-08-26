#include <stddef.h>
#include <stdint.h>

#if defined(__APPLE__)
#define VTA_CASE_SECTION(name, alignment) __attribute__((aligned(alignment)))
#else
#define VTA_CASE_SECTION(name, alignment) \
    __attribute__((section(name), aligned(alignment)))
#endif

/* Generated VTA single-operator SoC case.
 * TARGET=tsim, HW_VER=0.0.2, LOG_BUS_WIDTH=6
 * Each 128-bit instruction is stored as two little-endian uint64_t words.
 * Include this data header from this case's C file only.
 */

uint64_t dense_gemm_bias_clip_insn[]
    VTA_CASE_SECTION(".insn_data", 1024) = {
    0x0000000000000000ULL, 0x0000000100010001ULL, 0x00020008002000a2ULL, 0x0000000000000000ULL,
    0x0000000000000110ULL, 0x0000000100010001ULL, 0x00000000000000c0ULL, 0x0000000100010001ULL,
    0x0000000004000408ULL, 0x0000000100010001ULL, 0x0002000800400122ULL, 0x0000000000000000ULL,
    0x0000000004000110ULL, 0x0000000100010001ULL, 0x00000000040000c0ULL, 0x0000000100010001ULL,
    0x000200080040012aULL, 0x0000000000000000ULL, 0x0000000000000580ULL, 0x0000000100010001ULL,
    0x0000000008000800ULL, 0x0000000100010001ULL, 0x0002000800600204ULL, 0x0000200000000000ULL,
    0x000000000c000c00ULL, 0x0000000100010001ULL, 0x0002000800800304ULL, 0x001f800000000000ULL,
    0x0000000010001000ULL, 0x0000000100010001ULL, 0x0002000800a00444ULL, 0xffe0900000000000ULL,
    0x0000000000000229ULL, 0x0000000100010001ULL, 0x0000000000000110ULL, 0x0000000000000000ULL,
    0x0000000000000140ULL, 0x0000000000000000ULL, 0x0000000000000018ULL, 0x0000000000000000ULL,
    0x0000000000000003ULL, 0x0000000000000000ULL,
};

uint32_t dense_gemm_bias_clip_uop[]
    VTA_CASE_SECTION(".uop_data", 64) = {
    0x00000000U, 0x00000000U, 0x00000800U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
};

uint64_t dense_gemm_bias_clip_inp[]
    VTA_CASE_SECTION(".inp_data", 64) = {
    0x0300fc0201fefd04ULL, 0x01fdfcfffefffdfcULL, 0xfdfe010401040400ULL, 0xfefdfc02fcfcff02ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
};

uint64_t dense_gemm_bias_clip_wgt[]
    VTA_CASE_SECTION(".wgt_data", 512) = {
    0x020002fdfd040000ULL, 0x0202fcff040201feULL, 0x01fd020302fdfe03ULL, 0x04feff04fcfcfd03ULL,
    0x040404030304feffULL, 0xfd030001fcff00fdULL, 0xfc0004fd03fd00ffULL, 0x01010200fcffff01ULL,
    0x00ff02fefe0004fcULL, 0x02fefffdfeff02ffULL, 0x01000004fe020401ULL, 0xfdffff0204ffff02ULL,
    0xfffdfcfefeffff01ULL, 0x03ff0201ff04fc03ULL, 0xfd0202fc01fe0202ULL, 0x0301fe03fcfc04ffULL,
    0x04fffffcff0402fdULL, 0x0402ff00fdff0301ULL, 0x020202ff0401ffffULL, 0xfe010401fd030101ULL,
    0x02fe020401000304ULL, 0x02fd01020000fe03ULL, 0x01fdff0403ffff04ULL, 0x020303fffd04fefdULL,
    0x00fdfdfe0104ff01ULL, 0x02010001fc01ff00ULL, 0x020000fcfdfcfc01ULL, 0xfc030404fe02fe02ULL,
    0x00fd0100fffdfd02ULL, 0x03020200040001fcULL, 0x010302040404fffdULL, 0x0202fd04fefcfc01ULL,
    0x020103fefd01fc02ULL, 0x0303feffff030404ULL, 0x00fd03010002fcfdULL, 0xfffe030302040203ULL,
    0xfc0403fffd03fc02ULL, 0x030000ff0203fd03ULL, 0x02010400fdfcfefeULL, 0xfffd030300fdfd04ULL,
    0x02ff02fd04fd04ffULL, 0x0304fdfdfc02ff02ULL, 0x01fcfd01fdfc00ffULL, 0x01fdfe02fefeff00ULL,
    0x040203fffdfffd03ULL, 0x03fcfc04fd04feffULL, 0x03fe0303fefdfe02ULL, 0xfc0103fefffefe02ULL,
    0x03fd04fc04fffc00ULL, 0xfc03feff03ff03fcULL, 0x0204fe0204fcfdffULL, 0xfe03010004ff0102ULL,
    0xfcfdfe0400fefe03ULL, 0xfefffffefe02fefeULL, 0xfeff04fcfe010102ULL, 0x0201fd00feff0304ULL,
    0xffff040200fcfe01ULL, 0xfe02fcfe0302ff03ULL, 0x030100fdfd04fd04ULL, 0x01fd02feffffffffULL,
    0x02fcff01fcfefe03ULL, 0x040103fdfdfc0302ULL, 0xfe0404fffffffdfdULL, 0xfcfeff03010101fdULL,
};

uint64_t dense_gemm_bias_clip_acc[]
    VTA_CASE_SECTION(".acc_data", 64) = {
    0xfffffffdffffffefULL, 0xfffffff0fffffff2ULL, 0xfffffff3fffffff8ULL, 0xfffffff6fffffffaULL,
    0xfffffff6fffffff5ULL, 0xffffffeefffffff2ULL, 0xffffffeefffffffeULL, 0xffffffeffffffff9ULL,
};

uint64_t dense_gemm_bias_clip_out[]
    VTA_CASE_SECTION(".out_data", 64) = {
    0x0000000000000000ULL, 0x0000000000000092ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
};

const uint64_t dense_gemm_bias_clip_expected[]
    VTA_CASE_SECTION(".ref_data", 8) = {
    0xe8fbfde0e0e21ee0ULL, 0xeff6e0041f14e0e0ULL,
};

#define DENSE_GEMM_BIAS_CLIP_INSN_COUNT \
    ((uint32_t)(sizeof(dense_gemm_bias_clip_insn) / 16U))
#define DENSE_GEMM_BIAS_CLIP_EXPECTED_SIZE \
    ((size_t)sizeof(dense_gemm_bias_clip_expected))

_Static_assert(sizeof(dense_gemm_bias_clip_insn) % 16U == 0U,
               "VTA instruction array is not a multiple of 128 bits");
_Static_assert(sizeof(dense_gemm_bias_clip_out) >= sizeof(dense_gemm_bias_clip_expected),
               "VTA output buffer is smaller than expected output");
