#!/bin/zsh
set -eu

soc_dir=${0:A:h:h}
workspace_dir=${soc_dir:h:h}
conda_env_dir=${CONDA_ENV_DIR:-/Users/xdl/tools/miniforge3-26.1.1-3/envs/tvm_py310}
riscv_dir=${RISCV_TOOLCHAIN_DIR:-${workspace_dir}/tools/xpack-riscv-none-elf-gcc-15.2.0-1}
build_dir=${SOC_SIM_BUILD_DIR:-${soc_dir}/build/sim}
config=${SOC_CONFIG:-${soc_dir}/config/sim_default.json}
sdk_cxx=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1

if [[ $(uname -s) != Darwin ]]; then
  print -u2 "sim_run_macos.sh only supports macOS"
  exit 2
fi
if [[ ! -x ${soc_dir}/.venv/bin/python ]]; then
  print -u2 "missing ${soc_dir}/.venv; create it with the tvm_py310 Python first"
  exit 2
fi
if [[ ! -x ${riscv_dir}/bin/riscv-none-elf-gcc ]]; then
  print -u2 "missing RISC-V toolchain: ${riscv_dir}"
  exit 2
fi

cd ${soc_dir}
${soc_dir}/.venv/bin/python tools/patch_litex_sim_macos.py --venv ${soc_dir}/.venv

export PATH=${soc_dir}/.venv/bin:${riscv_dir}/bin:${conda_env_dir}/bin:/usr/bin:/bin:/usr/sbin:/sbin
export CFLAGS=-I${conda_env_dir}/include
export LDFLAGS=-L${conda_env_dir}/lib
export DYLD_LIBRARY_PATH=${conda_env_dir}/lib

# LiteX first generates and cross-compiles all artifacts.  The native C++ phase
# can fail because Conda Verilator records an obsolete prefixed clang; Vsim.mk
# is nevertheless complete and is rebuilt below with the system toolchain.
set +e
${soc_dir}/.venv/bin/python litex/vta_soc.py \
  --config ${config} --build-dir ${build_dir} --sim --build
litex_status=$?
set -e
if [[ ! -f ${build_dir}/gateware/obj_dir/Vsim.mk ]]; then
  print -u2 "LiteX generation failed before Vsim.mk was produced (status ${litex_status})"
  exit ${litex_status}
fi

make -C ${build_dir}/gateware/obj_dir -f Vsim.mk -j4 \
  CXX=/usr/bin/clang++ LINK=/usr/bin/clang++ AR=/usr/bin/ar \
  "VM_USER_CFLAGS=-nostdinc++ -isystem ${sdk_cxx} -I${conda_env_dir}/include -I/opt/homebrew/include -Wall -O0 -I${soc_dir}/.venv/lib/python3.10/site-packages/litex/build/sim/core" \
  "VM_USER_LDLIBS=libdylib.o modules.o pads.o parse.o sim.o -L${conda_env_dir}/lib -lpthread -ljson-c -lz -lm -lc++ -ldl -levent"

cd ${build_dir}/gateware
exec ./obj_dir/Vsim sim_config.js
