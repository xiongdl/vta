#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
workspace_dir="$(cd "${repo_dir}/.." && pwd)"
tvm_source_dir="${TVM_SOURCE_DIR:-${workspace_dir}/tvm}"
tvm_build_dir="${TVM_BUILD_DIR:-${tvm_source_dir}/build}"
config_json="${VTA_FSIM_CONFIG:-${repo_dir}/config/vta_config_fsim.json}"
build_dir="${repo_dir}/build/fsim"
jobs="${VTA_BUILD_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"
tvm_status_before="$(git -C "${tvm_source_dir}" status --short)"

export TVM_SOURCE_DIR="${tvm_source_dir}"
export TVM_BUILD_DIR="${tvm_build_dir}"
export VTA_CONFIG="${config_json}"
export VTA_LIBRARY_PATH="${repo_dir}/build/lib"
export VTA_HW_PATH="${repo_dir}"
export TVM_LIBRARY_PATH="${tvm_build_dir}"
export PYTHONPATH="${repo_dir}/python:${tvm_source_dir}/python${PYTHONPATH:+:${PYTHONPATH}}"

"${repo_dir}/scripts/build_simulators.sh" sim

python -c 'import vta; assert vta.get_env().TARGET == "sim"'
python "${repo_dir}/scripts/verify_isolation.py"
pytest -q \
  "${repo_dir}/tests/python/test_requantize.py" \
  "${repo_dir}/tests/python/test_relay_partition.py" \
  "${repo_dir}/tests/python/test_fsim_vector_add.py" \
  "${repo_dir}/tests/python/test_relay_dense_fsim.py" \
  "${repo_dir}/tests/python/test_relay_conv2d_fsim.py" \
  "${repo_dir}/tests/python/test_execution_plan_fsim.py" \
  "${repo_dir}/tests/python/test_external_codegen_fsim.py"

if [[ "$(git -C "${tvm_source_dir}" status --short)" != "${tvm_status_before}" ]]; then
  echo "standalone VTA regression changed the TVM source tree" >&2
  git -C "${tvm_source_dir}" status --short >&2
  exit 1
fi
