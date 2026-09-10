/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace vta {

namespace {

constexpr const char* kRelayToTIRBridge = "vta.relay._relay_to_tir";

}  // namespace

transform::Pass RelayToTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [](IRModule mod, transform::PassContext) {
        const runtime::PackedFunc* relay_to_tir = runtime::Registry::Get(kRelayToTIRBridge);
        ICHECK(relay_to_tir) << "VTA RelayToTIR bridge is unavailable; import the full vta package "
                               "before Relay lowering";
        mod = (*relay_to_tir)(mod);
        return mod;
      };
  return transform::CreateModulePass(pass_func, 0, "vta.RelayToTIR", {});
}

}  // namespace vta
}  // namespace tvm
