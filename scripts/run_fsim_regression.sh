#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
workspace_dir="$(cd "${repo_dir}/.." && pwd)"
tvm_source_dir="${TVM_SOURCE_DIR:-${workspace_dir}/tvm}"
tvm_build_dir="${TVM_BUILD_DIR:-${tvm_source_dir}/build}"
config_json="${VTA_CONFIG:-${repo_dir}/config/vta_config_fsim.json}"
build_dir="${VTA_FSIM_BUILD_DIR:-${repo_dir}/build-fsim}"
jobs="${VTA_BUILD_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"

export TVM_SOURCE_DIR="${tvm_source_dir}"
export TVM_BUILD_DIR="${tvm_build_dir}"
export VTA_CONFIG="${config_json}"
export VTA_LIBRARY_PATH="${build_dir}"
export VTA_HW_PATH="${repo_dir}"
export TVM_LIBRARY_PATH="${tvm_build_dir}"
export PYTHONPATH="${repo_dir}/python:${tvm_source_dir}/python${PYTHONPATH:+:${PYTHONPATH}}"

cmake -S "${repo_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${VTA_HOST_CC:-/usr/bin/clang}" \
  -DCMAKE_CXX_COMPILER="${VTA_HOST_CXX:-/usr/bin/clang++}" \
  -DTVM_SOURCE_DIR="${tvm_source_dir}" \
  -DTVM_BUILD_DIR="${tvm_build_dir}" \
  -DVTA_CONFIG="${config_json}" \
  -DVTA_RUNTIME=sim
cmake --build "${build_dir}" --parallel "${jobs}"

python -c 'import vta; assert vta.get_env().TARGET == "sim"'
pytest -q \
  "${repo_dir}/tests/python/test_relay_partition.py" \
  "${repo_dir}/tests/python/test_fsim_vector_add.py" \
  "${repo_dir}/tests/python/test_relay_dense_fsim.py" \
  "${repo_dir}/tests/python/test_relay_conv2d_fsim.py" \
  "${repo_dir}/tests/python/test_execution_plan_fsim.py" \
  "${repo_dir}/tests/python/test_external_codegen_fsim.py"

if [[ -n "$(git -C "${tvm_source_dir}" status --short)" ]]; then
  echo "TVM source tree is not clean after standalone VTA regression" >&2
  git -C "${tvm_source_dir}" status --short >&2
  exit 1
fi
