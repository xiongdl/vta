<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

VTA Hardware Design Stack
=========================
[![Build Status](https://ci.tlcpack.ai/job/tvm-vta/job/main/badge/icon)](https://ci.tlcpack.ai/job/tvm-vta/job/main/)

VTA (versatile tensor accelerator) is an open-source deep learning accelerator complemented with an end-to-end TVM-based compiler stack.

The key features of VTA include:

- Generic, modular, open-source hardware
  - Streamlined workflow to deploy to FPGAs.
  - Simulator support to prototype compilation passes on regular workstations.
- Driver and JIT runtime for both simulator and FPGA hardware back-end.
- End-to-end TVM stack integration
  - Direct optimization and deployment of models from deep learning frameworks via TVM.
  - Customized and extensible TVM compiler back-end.
  - Flexible RPC support to ease deployment, and program FPGAs with the convenience of Python.

Capability-based Relay compilation
----------------------------------

Importing `vta` loads and validates the VTA target extension. Applications then
invoke capability-based partitioning explicitly before building for the modern
VTA Relay target. Unsupported operators remain on the LLVM host, and no graph
operator names or range controls are required:

```python
import tvm
import vta
from tvm import relay

env = vta.get_env()
partitioned = vta.relay.partition_for_vta(mod, params=params)

with vta.build_config():
    factory = relay.build(
        partitioned,
        target=tvm.target.Target("vta", host=env.target_host),
    )
```

The outer `vta.build_config()` is required so host functions can safely access
VTA device buffers. The target extension lowers every outlined VTA function
through its RelayToTIR and TIRToRuntime hooks, and the resulting standard host
module can be exported and loaded through TVM's normal runtime APIs.

Modern Relay compilation uses `tvm.target.Target("vta")`. This is intentionally
distinct from the preserved low-level `env.target` and `tvm.target.vta()`
helpers, which describe an `ext_dev -device=vta` target for direct TE/TIR,
instruction, and hardware-development workflows. Those low-level APIs remain
available through `vta.build_config()`, `vta.build()`, and `vta.lower()`.
