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

"""Capability-based Relay partitioning for the VTA external compiler."""

from collections.abc import Mapping

import tvm
from tvm import relay
from tvm.relay.build_module import bind_params_by_name

from ..environment import get_env
from .contract import COMPILER_NAME, VTACompilerConfig
from .patterns import pattern_table


def _validate_inputs(mod, params, mod_name):
    if not isinstance(mod, tvm.IRModule):
        raise TypeError("mod must be a tvm.IRModule")
    if params is not None and not isinstance(params, Mapping):
        raise TypeError("params must be a mapping or None")
    if (
        not isinstance(mod_name, str)
        or not mod_name
        or any(ord(character) < 32 or ord(character) == 127 for character in mod_name)
    ):
        raise ValueError("mod_name must be a non-empty string without control characters")


def _partition_pipeline(config, mod_name):
    return tvm.transform.Sequential(
        [
            relay.transform.InferType(),
            relay.transform.MergeComposite(pattern_table(config)),
            relay.transform.AnnotateTarget(COMPILER_NAME),
            relay.transform.MergeCompilerRegions(),
            relay.transform.PartitionGraph(mod_name=mod_name),
            relay.transform.InferType(),
        ]
    )


def partition_for_vta(mod, params=None, mod_name="default"):
    """Partition supported Relay regions for the VTA external compiler."""
    _validate_inputs(mod, params, mod_name)
    if params:
        try:
            main = mod["main"]
        except tvm.error.TVMError as err:
            raise ValueError("mod must contain a main function when params are provided") from err
        mod["main"] = bind_params_by_name(main, dict(params))

    config = VTACompilerConfig.from_env(get_env())
    return _partition_pipeline(config, mod_name)(mod)
