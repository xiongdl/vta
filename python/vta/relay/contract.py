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

"""Stable compiler and execution-target contract for VTA BYOC."""

from dataclasses import dataclass

import tvm


COMPILER_NAME = "vta"
EXTERNAL_COMPILER = "relay.ext.vta"


def _positive_integer(env, name):
    value = getattr(env, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dtype(env, name):
    value = getattr(env, name, None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty dtype string")
    try:
        tvm.DataType(value)
    except ValueError as err:
        raise ValueError(f"{name} must be a valid TVM dtype") from err
    return value


@dataclass(frozen=True)
class VTACompilerConfig:
    """Immutable compilation-relevant snapshot of a VTA environment."""

    batch: int
    block_in: int
    block_out: int
    input_dtype: str
    weight_dtype: str
    accumulator_dtype: str
    output_dtype: str
    target: str
    host_target: str
    model: str
    device_type: int

    @classmethod
    def from_env(cls, env):
        """Create and validate a compiler snapshot from a VTA environment."""
        if env is None:
            raise TypeError("env must be a VTA environment")

        target = getattr(env, "target", None)
        if not isinstance(target, tvm.target.Target):
            raise TypeError("target must be a tvm.target.Target")
        if target.kind.name != "ext_dev" or target.device_name != COMPILER_NAME:
            raise ValueError("target must use the ext_dev kind with device=vta")

        host_target = getattr(env, "target_host", None)
        try:
            tvm.target.Target(host_target)
        except (TypeError, ValueError, tvm.error.TVMError) as err:
            raise ValueError("target_host must define a valid TVM target") from err

        model = getattr(env, "MODEL", None)
        if not isinstance(model, str) or not model:
            raise ValueError("MODEL must be a non-empty string")

        return cls(
            batch=_positive_integer(env, "BATCH"),
            block_in=_positive_integer(env, "BLOCK_IN"),
            block_out=_positive_integer(env, "BLOCK_OUT"),
            input_dtype=_dtype(env, "inp_dtype"),
            weight_dtype=_dtype(env, "wgt_dtype"),
            accumulator_dtype=_dtype(env, "acc_dtype"),
            output_dtype=_dtype(env, "out_dtype"),
            target=str(target),
            host_target=str(host_target),
            model=model,
            device_type=target.get_target_device_type(),
        )
