#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vta/hw_spec.h>

static VTAGenericInsn mem_insn(int opcode, int type, int size,
                               int pop_prev, int pop_next,
                               int push_prev, int push_next) {
  VTAInsn value{};
  value.mem.opcode = opcode;
  value.mem.memory_type = type;
  value.mem.y_size = 1;
  value.mem.x_size = size;
  value.mem.x_stride = size;
  value.mem.pop_prev_dep = pop_prev;
  value.mem.pop_next_dep = pop_next;
  value.mem.push_prev_dep = push_prev;
  value.mem.push_next_dep = push_next;
  return value.generic;
}

static VTAGenericInsn gemm_insn() {
  VTAInsn value{};
  value.gemm.opcode = VTA_OPCODE_GEMM;
  value.gemm.pop_prev_dep = 1;
  value.gemm.push_next_dep = 1;
  value.gemm.uop_end = 1;
  value.gemm.iter_out = 1;
  value.gemm.iter_in = 1;
  return value.generic;
}

static VTAGenericInsn finish_insn() {
  VTAInsn value{};
  value.gemm.opcode = VTA_OPCODE_FINISH;
  value.gemm.pop_next_dep = 1;
  return value.generic;
}

static void write_file(const char* directory, const char* name,
                       const void* data, size_t size) {
  char path[1024];
  std::snprintf(path, sizeof(path), "%s/%s", directory, name);
  FILE* stream = std::fopen(path, "wb");
  if (stream == nullptr || std::fwrite(data, 1, size, stream) != size ||
      std::fclose(stream) != 0) {
    std::perror(path);
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  if (sizeof(VTAGenericInsn) != 16 || sizeof(VTAUop) != 4) return 3;

  VTAGenericInsn insns[7] = {
      mem_insn(VTA_OPCODE_LOAD, VTA_MEM_ID_UOP, 1, 0, 0, 0, 0),
      mem_insn(VTA_OPCODE_LOAD, VTA_MEM_ID_ACC, 1, 0, 0, 1, 0),
      mem_insn(VTA_OPCODE_LOAD, VTA_MEM_ID_WGT, 1, 0, 1, 0, 0),
      mem_insn(VTA_OPCODE_LOAD, VTA_MEM_ID_INP, 1, 0, 0, 0, 1),
      gemm_insn(),
      mem_insn(VTA_OPCODE_STORE, VTA_MEM_ID_OUT, 1, 1, 0, 1, 0),
      finish_insn(),
  };
  VTAUop uop{};
  int8_t input[VTA_BATCH * VTA_BLOCK_IN];
  int8_t weight[VTA_BLOCK_OUT * VTA_BLOCK_IN];
  int32_t bias[VTA_BATCH * VTA_BLOCK_OUT]{};
  int8_t output[VTA_BATCH * VTA_BLOCK_OUT];
  std::memset(input, 1, sizeof(input));
  std::memset(weight, 1, sizeof(weight));
  std::memset(output, 0xa5, sizeof(output));

  write_file(argv[1], "insn.bin", insns, sizeof(insns));
  write_file(argv[1], "uop.bin", &uop, sizeof(uop));
  write_file(argv[1], "input.bin", input, sizeof(input));
  write_file(argv[1], "weight.bin", weight, sizeof(weight));
  write_file(argv[1], "bias.bin", bias, sizeof(bias));
  write_file(argv[1], "output.bin", output, sizeof(output));
  return 0;
}
