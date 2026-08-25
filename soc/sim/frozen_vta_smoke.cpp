#include <cstdint>
#include <cstdio>

#include "VVTAShellAPB.h"
#include "verilated.h"

namespace {

vluint64_t sim_time = 0;

void tick(VVTAShellAPB& dut) {
  dut.clock = 0;
  dut.eval();
  ++sim_time;
  dut.clock = 1;
  dut.eval();
  ++sim_time;
}

void apb_write(VVTAShellAPB& dut, uint16_t address, uint32_t value) {
  dut.io_host_paddr = address;
  dut.io_host_pwrite = 1;
  dut.io_host_pwdata = value;
  dut.io_host_pstrb = 0xf;
  dut.io_host_psel = 1;
  dut.io_host_penable = 0;
  tick(dut);
  dut.io_host_penable = 1;
  tick(dut);
  if (!dut.io_host_pready || dut.io_host_pslverr) {
    std::fprintf(stderr, "APB write failed at 0x%04x\n", address);
    std::exit(2);
  }
  dut.io_host_psel = 0;
  dut.io_host_penable = 0;
  tick(dut);
}

}  // namespace

double sc_time_stamp() { return static_cast<double>(sim_time); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  VVTAShellAPB dut;

  dut.reset = 1;
  dut.io_host_paddr = 0;
  dut.io_host_psel = 0;
  dut.io_host_penable = 0;
  dut.io_host_pwrite = 0;
  dut.io_host_pwdata = 0;
  dut.io_host_pstrb = 0;
  dut.io_host_pprot = 0;
  dut.io_mem_aw_ready = 1;
  dut.io_mem_w_ready = 1;
  dut.io_mem_b_valid = 0;
  dut.io_mem_b_bits_resp = 0;
  dut.io_mem_b_bits_id = 0;
  dut.io_mem_b_bits_user = 0;
  dut.io_mem_ar_ready = 1;
  dut.io_mem_r_valid = 0;
  dut.io_mem_r_bits_data = 0;
  dut.io_mem_r_bits_resp = 0;
  dut.io_mem_r_bits_last = 0;
  dut.io_mem_r_bits_id = 0;
  dut.io_mem_r_bits_user = 0;

  for (int i = 0; i < 8; ++i) tick(dut);
  dut.reset = 0;
  for (int i = 0; i < 4; ++i) tick(dut);

  constexpr uint32_t kInsnAddress = 0x40010000;
  apb_write(dut, 0x08, 7);
  apb_write(dut, 0x0c, kInsnAddress);
  apb_write(dut, 0x00, 1);

  bool saw_ar = false;
  for (int cycle = 0; cycle < 64; ++cycle) {
    dut.clock = 0;
    dut.eval();
    if (dut.io_mem_ar_valid) {
      saw_ar = true;
      if (dut.io_mem_ar_bits_addr != kInsnAddress) {
        std::fprintf(stderr, "wrong AXI AR address: 0x%08x\n",
                     dut.io_mem_ar_bits_addr);
        return 3;
      }
      std::printf("frozen VTA fetch PASS: AR addr=0x%08x len=%u size=%u\n",
                  dut.io_mem_ar_bits_addr, dut.io_mem_ar_bits_len,
                  dut.io_mem_ar_bits_size);
      break;
    }
    tick(dut);
  }
  dut.final();
  if (!saw_ar) {
    std::fprintf(stderr, "frozen VTA fetch FAIL: no AXI ARVALID after launch\n");
    return 1;
  }
  return 0;
}
