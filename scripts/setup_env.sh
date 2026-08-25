#!/usr/bin/env bash

if [[ -n "${ZSH_VERSION:-}" ]]; then
  source_path="${(%):-%N}"
else
  source_path="${BASH_SOURCE[0]}"
fi

vta_repo_dir="$(cd "$(dirname "${source_path}")/.." && pwd)"
workspace_dir="$(cd "${vta_repo_dir}/.." && pwd)"
export VTA_HW_PATH="${vta_repo_dir}"
export VTA_LIBRARY_PATH="${vta_repo_dir}/build/lib"
export VTA_CONFIG="${VTA_CONFIG:-${vta_repo_dir}/config/vta_config.json}"
export TVM_LIBRARY_PATH="${workspace_dir}/tvm/build"
export PYTHONPATH="${vta_repo_dir}/python:${workspace_dir}/tvm/python"

case ":${PYTHONPATH}:" in
  *":${workspace_dir}/tvm/vta/python:"*)
    echo "tvm/vta/python is forbidden by the project isolation rule" >&2
    return 2
    ;;
esac
