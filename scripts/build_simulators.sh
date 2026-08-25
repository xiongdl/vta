#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
tvm_source_dir="${TVM_SOURCE_DIR:-${repo_dir}/../tvm}"
tvm_build_dir="${TVM_BUILD_DIR:-${tvm_source_dir}/build}"
fsim_config="${VTA_FSIM_CONFIG:-${repo_dir}/config/vta_config_fsim.json}"
tsim_config="${VTA_TSIM_CONFIG:-${repo_dir}/config/vta_config.json}"
build_root="${VTA_BUILD_ROOT:-${repo_dir}/build}"
runtime="${1:-all}"
jobs="${VTA_BUILD_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"
cc_bin="${VTA_HOST_CC:-/usr/bin/clang}"
cxx_bin="${VTA_HOST_CXX:-/usr/bin/clang++}"

build_runtime() {
  local kind="$1"
  local output_dir="${build_root}/${kind}"
  local config_json="${tsim_config}"
  local extra_args=("-DVTA_OUTPUT_NAME=")
  if [[ "${kind}" == "sim" ]]; then
    config_json="${fsim_config}"
  fi
  if [[ "${kind}" == "tsim" ]]; then
    local verilator_bin="${VERILATOR:-verilator}"
    local verilator_root
    verilator_root="$(${verilator_bin} -V | tr -d '\000' | awk -F= '/VERILATOR_ROOT/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}')"
    extra_args+=("-DVERILATOR_INC_DIR=${VERILATOR_INC_DIR:-${verilator_root}/include}")
  fi
  cmake -S "${repo_dir}" -B "${output_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${cc_bin}" \
    -DCMAKE_CXX_COMPILER="${cxx_bin}" \
    -DTVM_SOURCE_DIR="${tvm_source_dir}" \
    -DTVM_BUILD_DIR="${tvm_build_dir}" \
    -DPython3_EXECUTABLE="${PYTHON:-python}" \
    -DVTA_CONFIG="${config_json}" \
    -DVTA_RUNTIME="${kind}" \
    -DCMAKE_INSTALL_PREFIX="${build_root}" \
    "${extra_args[@]}"
  cmake --build "${output_dir}" --parallel "${jobs}"
  cmake --install "${output_dir}"
}

build_hardware() {
  local python_bin="${PYTHON:-python}"
  local verilator_bin="${VERILATOR:-verilator}"
  local verilator_root
  verilator_root="$(${verilator_bin} -V | tr -d '\000' | awk -F= '/VERILATOR_ROOT/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}')"
  export VTA_HW_PATH="${repo_dir}"
  export TVM_PATH="${tvm_source_dir}"
  export VERILATOR_INC_DIR="${VERILATOR_INC_DIR:-${verilator_root}/include}"
  export CC="${cc_bin}"
  export CXX="${cxx_bin}"
  if [[ -z "${JAVA_HOME:-}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/lib/jvm/bin/java" ]]; then
    export JAVA_HOME="${CONDA_PREFIX}/lib/jvm"
  fi
  while read -r key value; do
    export "${key}=${value}"
  done < <("${python_bin}" "${repo_dir}/scripts/tsim_config.py" "${tsim_config}")
  make -C "${repo_dir}/hardware/chisel" cleanall
  make -C "${repo_dir}/hardware/chisel" CONFIG=DefaultTSimConfig USE_THREADS=0 lib
  cmake -E make_directory "${build_root}/lib"
  if [[ -f "${build_root}/libvta_hw.dylib" ]]; then
    cmake -E copy_if_different "${build_root}/libvta_hw.dylib" "${build_root}/lib/libvta_hw.dylib"
  elif [[ -f "${build_root}/libvta_hw.so" ]]; then
    cmake -E copy_if_different "${build_root}/libvta_hw.so" "${build_root}/lib/libvta_hw.so"
  fi
}

case "${runtime}" in
  sim) build_runtime sim ;;
  tsim) build_runtime tsim; build_hardware ;;
  all) build_runtime sim; build_runtime tsim; build_hardware ;;
  *) echo "Usage: $0 [sim|tsim|all]" >&2; exit 2 ;;
esac
