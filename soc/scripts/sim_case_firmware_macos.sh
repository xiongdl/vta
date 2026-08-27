#!/bin/zsh
set -eu
gateware=${1:?usage: $0 build-dir/gateware}
simulator=${gateware}/obj_dir/Vsim
if [[ ! -x ${simulator} ]]; then
  print -u2 "case simulator not found: ${simulator}"
  print -u2 "build it first with: make case-firmware CASE_DIR=/path/to/case"
  exit 2
fi
soc_dir=${0:A:h:h}
python=${SOC_PYTHON:-${soc_dir}/.venv/bin/python}
conda_env_dir=${CONDA_ENV_DIR:-${CONDA_PREFIX:-}}
if [[ -z ${conda_env_dir} ]]; then
  conda_env_dir=$(${python} -c 'import sys; print(sys.base_prefix)')
fi
export DYLD_LIBRARY_PATH=${conda_env_dir}/lib
cd ${gateware}
exec ${python} ${soc_dir}/tools/run_case_firmware.py ./obj_dir/Vsim sim_config.js
