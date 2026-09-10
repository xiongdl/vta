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
#include <tvm/target/target.h>

namespace tvm {

using FTVMTIRToRuntime = runtime::TypedPackedFunc<runtime::Module(IRModule, Target)>;

namespace vta {

transform::Pass RelayToTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [](IRModule mod, transform::PassContext) { return mod; };
  return transform::CreateModulePass(pass_func, 0, "vta.RelayToTIR.Foundation", {});
}

runtime::Module TIRToRuntime(IRModule, Target) {
  LOG(FATAL) << "VTA TIRToRuntime is not implemented by the target-extension foundation";
  return runtime::Module();
}

}  // namespace vta

TVM_REGISTER_TARGET_KIND("vta", kDLExtDev)
    .set_attr<Bool>("use_device_api", Bool(true))
    .set_attr<relay::transform::FTVMRelayToTIR>(attr::kRelayToTIR, vta::RelayToTIR())
    .set_attr<FTVMTIRToRuntime>("TIRToRuntime", vta::TIRToRuntime);

}  // namespace tvm
