# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Contracts for the canonical, build-bound VTA ABI fingerprint."""

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


VTA_ROOT = Path(__file__).resolve().parents[3]
CONFIG_TOOL_PATH = VTA_ROOT / "config" / "vta_config.py"
DEFAULT_CONFIG_PATH = VTA_ROOT / "config" / "vta_config.json"
ABI_CONFIG_KEYS = (
    "LOG_INP_WIDTH",
    "LOG_WGT_WIDTH",
    "LOG_ACC_WIDTH",
    "LOG_BATCH",
    "LOG_BLOCK",
    "LOG_UOP_BUFF_SIZE",
    "LOG_INP_BUFF_SIZE",
    "LOG_WGT_BUFF_SIZE",
    "LOG_ACC_BUFF_SIZE",
)


def _load_config_tool():
    spec = importlib.util.spec_from_file_location("vta_config_tool", CONFIG_TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(path, config, *, indent=None, sort_keys=False):
    path.write_text(
        json.dumps(config, indent=indent, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def _run_config_tool(config_path, *arguments):
    return subprocess.run(
        [
            sys.executable,
            str(CONFIG_TOOL_PATH),
            "--use-cfg={}".format(config_path),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _fingerprint_from_cli(config_path):
    output = _run_config_tool(config_path, "--abi-fingerprint").stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{16}", output)
    return output


def _reference_fingerprint(definitions, schema_version):
    descriptor = {
        "definitions": list(definitions),
        "schema_version": schema_version,
    }
    canonical_json = json.dumps(
        descriptor,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(canonical_json).digest()[:8], "big")


@pytest.fixture
def default_config():
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_fingerprint_is_independent_of_json_order_format_path_and_timestamp(
    tmp_path, default_config
):
    compact_path = tmp_path / "compact" / "first.json"
    pretty_path = tmp_path / "elsewhere" / "second.json"
    compact_path.parent.mkdir()
    pretty_path.parent.mkdir()
    _write_config(compact_path, default_config)
    _write_config(
        pretty_path,
        dict(reversed(list(default_config.items()))),
        indent=4,
        sort_keys=True,
    )
    os.utime(compact_path, (1_600_000_000, 1_600_000_000))
    os.utime(pretty_path, (1_900_000_000, 1_900_000_000))

    assert _fingerprint_from_cli(compact_path) == _fingerprint_from_cli(pretty_path)


@pytest.mark.parametrize("config_key", ABI_CONFIG_KEYS)
def test_every_abi_config_field_changes_the_fingerprint(tmp_path, default_config, config_key):
    baseline_path = tmp_path / "baseline.json"
    changed_path = tmp_path / "changed.json"
    changed_config = dict(default_config)
    changed_config[config_key] += 1
    _write_config(baseline_path, default_config)
    _write_config(changed_path, changed_config)

    assert _fingerprint_from_cli(baseline_path) != _fingerprint_from_cli(changed_path)


def test_schema_version_changes_the_fingerprint(default_config):
    config_tool = _load_config_tool()
    definitions = config_tool.abi_definitions(default_config)

    first = config_tool.abi_fingerprint(definitions, schema_version=1)
    second = config_tool.abi_fingerprint(definitions, schema_version=2)

    assert first != second


def test_definition_order_does_not_change_the_fingerprint(default_config):
    config_tool = _load_config_tool()
    definitions = config_tool.abi_definitions(default_config)

    assert tuple(definitions) == tuple(sorted(definitions))
    assert config_tool.abi_fingerprint(definitions) == config_tool.abi_fingerprint(
        reversed(definitions)
    )


def test_every_normalized_definition_changes_the_fingerprint(default_config):
    config_tool = _load_config_tool()
    definitions = tuple(config_tool.abi_definitions(default_config))
    baseline = config_tool.abi_fingerprint(definitions)

    for index, definition in enumerate(definitions):
        name, value = definition.split("=", 1)
        changed = (
            definitions[:index]
            + ("{}={}1".format(name, value),)
            + definitions[index + 1 :]
        )
        assert config_tool.abi_fingerprint(changed) != baseline, name


def test_fingerprint_uses_the_canonical_descriptor_and_algorithm(default_config):
    config_tool = _load_config_tool()
    definitions = config_tool.abi_definitions(default_config)

    expected = _reference_fingerprint(definitions, config_tool.ABI_SCHEMA_VERSION)

    assert config_tool.abi_fingerprint(definitions) == expected


def test_generated_header_is_one_stable_compile_time_source(tmp_path, default_config):
    config_path = tmp_path / "config.json"
    header_path = tmp_path / "generated" / "vta" / "abi_config.h"
    _write_config(config_path, default_config, indent=2)

    _run_config_tool(config_path, "--abi-header={}".format(header_path))

    fingerprint = _fingerprint_from_cli(config_path)
    assert header_path.read_text(encoding="utf-8") == (
        "#ifndef VTA_ABI_CONFIG_H_\n"
        "#define VTA_ABI_CONFIG_H_\n"
        "\n"
        "#include <stdint.h>\n"
        "\n"
        "#define VTA_ABI_SCHEMA_VERSION 1\n"
        "#define VTA_ABI_FINGERPRINT UINT64_C(0x{})\n"
        "\n"
        "#endif  // VTA_ABI_CONFIG_H_\n".format(fingerprint)
    )


def test_generated_header_is_reproducible(tmp_path, default_config):
    config_path = tmp_path / "config.json"
    first_header = tmp_path / "first" / "vta" / "abi_config.h"
    second_header = tmp_path / "second" / "vta" / "abi_config.h"
    _write_config(config_path, default_config)

    _run_config_tool(config_path, "--abi-header={}".format(first_header))
    os.utime(config_path, (1_900_000_000, 1_900_000_000))
    _run_config_tool(config_path, "--abi-header={}".format(second_header))

    assert first_header.read_bytes() == second_header.read_bytes()
