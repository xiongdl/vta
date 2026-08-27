#!/bin/zsh
# Build a LiteX/Verilator executable without running it.
set -eu

soc_dir=${0:A:h:h}
build_dir=${1:?usage: $0 build-dir config [case-dir]}
config=${2:?usage: $0 build-dir config [case-dir]}
case_dir=${3:-}
python=${SOC_PYTHON:-${soc_dir}/.venv/bin/python}

if [[ ! -x ${python} ]]; then
  print -u2 "missing SoC Python environment: ${python}"
  print -u2 "create it once with the tvm_py310 Python and install soc/requirements.txt"
  exit 2
fi

args=(--config ${config:A} --build-dir ${build_dir:A} --sim --build)
if [[ -n ${case_dir} ]]; then
  args+=(--case-dir ${case_dir:A})
fi

cd ${soc_dir}
if [[ $(uname -s) != Darwin ]]; then
  exec ${python} litex/vta_soc.py ${args[@]}
fi

conda_env_dir=${CONDA_ENV_DIR:-${CONDA_PREFIX:-}}
if [[ -z ${conda_env_dir} ]]; then
  conda_env_dir=$(${python} -c 'import sys; print(sys.base_prefix)')
fi
riscv_dir=${RISCV_TOOLCHAIN_DIR:-${soc_dir:h:h}/tools/xpack-riscv-none-elf-gcc-15.2.0-1}
apple_cxx=$(xcrun --find clang++)
apple_ar=$(xcrun --find ar)
sdk_root=${SDKROOT:-$(xcrun --show-sdk-path)}
sdk_cxx_headers=${sdk_root}/usr/include/c++/v1
litex_sim_core=${soc_dir}/.venv/lib/python3.10/site-packages/litex/build/sim/core

if [[ ! -x ${riscv_dir}/bin/riscv-none-elf-gcc ]]; then
  print -u2 "missing RISC-V toolchain: ${riscv_dir}"
  exit 2
fi
for required in ${conda_env_dir}/bin/verilator ${apple_cxx} ${apple_ar}; do
  if [[ ! -x ${required} ]]; then
    print -u2 "missing tool in tvm_py310: ${required}"
    exit 2
  fi
done
if [[ ! -d ${sdk_cxx_headers} ]]; then
  print -u2 "missing Apple SDK libc++ headers: ${sdk_cxx_headers}"
  exit 2
fi

${python} tools/patch_litex_sim_macos.py --venv ${soc_dir}/.venv
export PATH=${soc_dir}/.venv/bin:${riscv_dir}/bin:${conda_env_dir}/bin:/usr/bin:/bin:/usr/sbin:/sbin
export CFLAGS=-I${conda_env_dir}/include
export LDFLAGS=-L${conda_env_dir}/lib
unset DYLD_LIBRARY_PATH 2>/dev/null || true
# Conda's compiler activation exports target-prefixed CC/CXX values. LiteX
# interprets CC as a boolean CPU option, so do not leak host compiler overrides
# into its RISC-V software-toolchain selection.
unset CC CXX CPP AR AS LD NM RANLIB STRIP OBJCOPY OBJDUMP \
  CLANG CLANGXX CC_FOR_BUILD CXX_FOR_BUILD CXXFLAGS DEBUG_CXXFLAGS \
  DEBUG_CFLAGS CMAKE_ARGS 2>/dev/null || true

# Let LiteX produce its complete Vsim.mk, then rebuild it with the Apple
# compiler/SDK matching this macOS host. Non-system libraries stay isolated in
# tvm_py310.
software_freshness_marker=$(mktemp /tmp/vta-soc-software.XXXXXX)
litex_log=$(mktemp /tmp/vta-soc-litex.XXXXXX)
trap 'rm -f ${software_freshness_marker} ${litex_log}' EXIT
set +e
${python} litex/vta_soc.py ${args[@]} >${litex_log} 2>&1
litex_status=$?
set -e
gateware=${build_dir:A}/gateware
makefile=${gateware}/obj_dir/Vsim.mk
software_bin=${build_dir:A}/software/bios/bios.bin
if [[ ! -f ${makefile} || ! -f ${software_bin} || ! ${software_bin} -nt ${software_freshness_marker} ]]; then
  cat ${litex_log}
  print -u2 "LiteX generation did not produce fresh software/Vsim.mk (status ${litex_status})"
  exit ${litex_status}
fi
print "LiteX software and simulator sources generated; compiling with the macOS host ABI."

make -C ${gateware}/obj_dir -f Vsim.mk -j4 \
  CXX=${apple_cxx} LINK=${apple_cxx} AR=${apple_ar} \
  "VM_USER_CFLAGS=-nostdinc++ -isystem ${sdk_cxx_headers} -isysroot ${sdk_root} -I${conda_env_dir}/include -I${litex_sim_core}" \
  "VM_USER_LDLIBS=libdylib.o modules.o pads.o parse.o sim.o -isysroot ${sdk_root} -L${conda_env_dir}/lib -lpthread -ljson-c -lz -lm -lc++ -ldl -levent"

test -x ${gateware}/obj_dir/Vsim
print "SoC simulator built: ${gateware}/obj_dir/Vsim"
