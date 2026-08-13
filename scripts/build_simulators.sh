#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
tvm_source_dir="${TVM_SOURCE_DIR:?Set TVM_SOURCE_DIR to the Apache TVM source tree}"
tvm_build_dir="${TVM_BUILD_DIR:?Set TVM_BUILD_DIR to the TVM build directory}"
config_json="${VTA_CONFIG:-${repo_dir}/config/vta_config.json}"
build_root="${VTA_BUILD_ROOT:-${repo_dir}/build-standalone}"
runtime="${1:-all}"
jobs="${VTA_BUILD_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"
cc_bin="${VTA_HOST_CC:-/usr/bin/clang}"
cxx_bin="${VTA_HOST_CXX:-/usr/bin/clang++}"

build_runtime() {
  local kind="$1"
  local output_dir="${build_root}-${kind}"
  local extra_args=()
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
    -DVTA_CONFIG="${config_json}" \
    -DVTA_RUNTIME="${kind}" \
    "${extra_args[@]}"
  cmake --build "${output_dir}" --parallel "${jobs}"
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
  while read -r key value; do
    export "${key}=${value}"
  done < <("${python_bin}" "${repo_dir}/scripts/tsim_config.py" "${config_json}")
  make -C "${repo_dir}/hardware/chisel" cleanall
  make -C "${repo_dir}/hardware/chisel" CONFIG=DefaultTSimConfig USE_THREADS=0 lib
}

case "${runtime}" in
  sim) build_runtime sim ;;
  tsim) build_runtime tsim; build_hardware ;;
  all) build_runtime sim; build_runtime tsim; build_hardware ;;
  *) echo "Usage: $0 [sim|tsim|all]" >&2; exit 2 ;;
esac
