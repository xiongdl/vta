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

# VTA Configuration

Each VTA runtime/hardware configuration is specified by vta_config.json file.
You can copy the vta_config.json to tvm project root and modify the configuration
before you type make.

The config is going to affect the behavior of python package as well as
the hardware runtime build.

For TSIM builds, `VTA_TSIM_MEM_READ_LATENCY` selects a non-negative number of
additional RTL read-response cycles. The same delay module is used by the AXI
and AHB memory shells, and the default is zero.

Set `"LOG_BUS_WIDTH": 5` in the TSIM JSON configuration so the software
runtime uses 32-bit beats. Build the matching Chisel design by selecting its
dedicated configuration:

```bash
make -C hardware/chisel CONFIG=DefaultSim32Config
```
