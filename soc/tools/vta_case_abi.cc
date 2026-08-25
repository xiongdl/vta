#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <vta/hw_spec.h>

static std::vector<VTAGenericInsn> read_insns(const char* path) {
  FILE* stream = std::fopen(path, "rb");
  if (!stream) { std::perror(path); std::exit(2); }
  std::fseek(stream, 0, SEEK_END);
  long size = std::ftell(stream);
  std::rewind(stream);
  if (size < 0 || size % sizeof(VTAGenericInsn) != 0) std::exit(3);
  std::vector<VTAGenericInsn> result(size / sizeof(VTAGenericInsn));
  if (std::fread(result.data(), 1, size, stream) != static_cast<size_t>(size)) std::exit(4);
  std::fclose(stream);
  return result;
}

int main(int argc, char** argv) {
  if (sizeof(VTAGenericInsn) != 16 || sizeof(VTAUop) != 4) return 10;
  if (argc == 3 && std::strcmp(argv[1], "inspect") == 0) {
    auto insns = read_insns(argv[2]);
    for (size_t i = 0; i < insns.size(); ++i) {
      VTAInsn value{};
      value.generic = insns[i];
      std::printf("%zu %llu %llu %llu %llu %llu %llu\n", i,
          static_cast<unsigned long long>(value.mem.opcode),
          static_cast<unsigned long long>(value.mem.memory_type),
          static_cast<unsigned long long>(value.mem.dram_base),
          static_cast<unsigned long long>(value.mem.y_size),
          static_cast<unsigned long long>(value.mem.x_size),
          static_cast<unsigned long long>(value.mem.x_stride));
    }
    return 0;
  }
  if (argc == 5 && std::strcmp(argv[1], "patch") == 0) {
    auto insns = read_insns(argv[2]);
    FILE* patches = std::fopen(argv[4], "r");
    if (!patches) { std::perror(argv[4]); return 5; }
    size_t index;
    unsigned long long dram_base;
    while (std::fscanf(patches, "%zu %llu", &index, &dram_base) == 2) {
      if (index >= insns.size()) return 6;
      VTAInsn value{};
      value.generic = insns[index];
      value.mem.dram_base = dram_base;
      insns[index] = value.generic;
    }
    std::fclose(patches);
    FILE* output = std::fopen(argv[3], "wb");
    if (!output) { std::perror(argv[3]); return 7; }
    size_t bytes = insns.size() * sizeof(VTAGenericInsn);
    if (std::fwrite(insns.data(), 1, bytes, output) != bytes || std::fclose(output) != 0) return 8;
    return 0;
  }
  std::fprintf(stderr, "usage: %s inspect INSN | patch INSN OUT PATCHES\n", argv[0]);
  return 1;
}
