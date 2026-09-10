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

#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cstdint>
#include <string>
#include <unordered_set>

namespace tvm {
namespace vta {
namespace {

constexpr const char* kLegacyTEScheduleAttr = "from_legacy_te_schedule";

[[noreturn]] void FailValidation(const std::string& message) {
  TVMAPISetLastError(message.c_str());
  throw runtime::EnvErrorAlreadySet(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) {
    FailValidation(message);
  }
}

bool IsLLVMTarget(const Target& target) { return target->kind->name == "llvm"; }

bool IsRawVTAFunctionTarget(const Target& target) {
  return target->kind->name == "vta" || target->HasKey("vta");
}

class RuntimeCallValidator : public tir::StmtExprVisitor {
 public:
  explicit RuntimeCallValidator(std::string symbol) : symbol_(std::move(symbol)) {}

  void Validate(const tir::Stmt& body) {
    VisitStmt(body);
    Require(has_vta_activity_,
            "VTA PrimFunc " + symbol_ + " does not contain a recognized VTA runtime call");
  }

 private:
  void VisitExpr_(const tir::CallNode* call) final {
    if (call->op.same_as(tir::builtin::call_extern())) {
      Require(call->args.size() > 0,
              "VTA PrimFunc " + symbol_ + " contains a malformed runtime call");
      const auto* name = call->args[0].as<tir::StringImmNode>();
      Require(name != nullptr, "VTA PrimFunc " + symbol_ + " contains a malformed runtime call");
      static const std::unordered_set<std::string> kRuntimeCalls = {
          "VTABufferCPUPtr", "VTADepPop",        "VTADepPush",      "VTALoadBuffer2D",
          "VTASetDebugMode", "VTAStoreBuffer2D", "VTASynchronize",  "VTATLSCommandHandle",
          "VTAUopLoopBegin", "VTAUopLoopEnd",    "VTAUopPush"};
      Require(kRuntimeCalls.count(name->value),
              "VTA PrimFunc " + symbol_ + " contains unsupported runtime call " +
                  std::string(name->value));
      has_vta_activity_ = true;
    } else if (const auto* op = call->op.as<OpNode>()) {
      static const std::unordered_set<std::string> kVTAOps = {
          "tir.vta.command_handle", "tir.vta.coproc_sync", "tir.vta.coproc_dep_push",
          "tir.vta.coproc_dep_pop", "tir.vta.uop_push"};
      std::string op_name = op->name;
      if (op_name.rfind("tir.vta.", 0) == 0) {
        Require(kVTAOps.count(op_name),
                "VTA PrimFunc " + symbol_ + " contains unsupported runtime call " + op_name);
        has_vta_activity_ = true;
      }
    }
    tir::StmtExprVisitor::VisitExpr_(call);
  }

  std::string symbol_;
  bool has_vta_activity_{false};
};

void ValidateModule(const IRModule& mod, const Target& target) {
  Require(target.defined() && target->kind->name == "vta",
          "VTA TIRToRuntime requires a vta target");
  Optional<Target> host = target->GetHost();
  Require(host.defined() && IsLLVMTarget(host.value()),
          "VTA TIRToRuntime requires an LLVM host target");
  Require(mod->functions.size() > 0, "VTA TIRToRuntime does not accept an empty module");

  std::unordered_set<std::string> symbols;
  for (const auto& [global_var, base_func] : mod->functions) {
    const auto* prim_func_node = base_func.as<tir::PrimFuncNode>();
    Require(prim_func_node != nullptr,
            "VTA TIRToRuntime requires every function to be a PrimFunc; " +
                std::string(global_var->name_hint) + " is not a PrimFunc");
    tir::PrimFunc prim_func = GetRef<tir::PrimFunc>(prim_func_node);
    Optional<String> global_symbol = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    if (!global_symbol.defined()) {
      FailValidation("VTA PrimFunc " + std::string(global_var->name_hint) +
                     " is missing global_symbol");
    }
    std::string symbol = global_symbol.value();
    Require(symbols.insert(symbol).second,
            "VTA module contains duplicate global_symbol " + symbol);
  }

  for (const auto& [global_var, base_func] : mod->functions) {
    const auto* prim_func_node = base_func.as<tir::PrimFuncNode>();
    tir::PrimFunc prim_func = GetRef<tir::PrimFunc>(prim_func_node);
    Optional<String> global_symbol = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    std::string symbol = global_symbol.value();
    Require(global_var->name_hint == symbol,
            "VTA PrimFunc " + std::string(global_var->name_hint) +
                " has mismatched global_symbol " + symbol);

    Optional<Target> function_target = prim_func->GetAttr<Target>(tvm::attr::kTarget);
    Require(function_target.defined(), "VTA PrimFunc " + symbol + " is missing target");
    auto calling_conv = prim_func->GetAttr<Integer>(tvm::attr::kCallingConv);
    bool is_packed = calling_conv.defined() &&
                     calling_conv.value()->value == static_cast<int>(CallingConv::kCPackedFunc);
    if (is_packed) {
      Require(IsLLVMTarget(function_target.value()),
              "VTA PrimFunc " + symbol + " has invalid packed target");
    } else {
      Require(IsRawVTAFunctionTarget(function_target.value()),
              "VTA PrimFunc " + symbol + " has invalid target");
      Optional<Target> function_host = function_target.value()->GetHost();
      Require(function_host.defined() && IsLLVMTarget(function_host.value()),
              "VTA PrimFunc " + symbol + " target requires an LLVM host");
    }

    RuntimeCallValidator(symbol).Validate(prim_func->body);
  }
}

IRModule ForceFlattenExternalBuffers(IRModule mod) {
  mod = mod->ShallowCopy();
  for (const auto& [global_var, base_func] : mod->functions) {
    tir::PrimFunc prim_func = Downcast<tir::PrimFunc>(base_func);
    prim_func = WithoutAttr(std::move(prim_func), kLegacyTEScheduleAttr);
    if (prim_func->HasNonzeroAttr("vta.route_to_runtime")) {
      Target routed_target = prim_func->GetAttr<Target>(tvm::attr::kTarget).value();
      Target host = routed_target->GetHost().value();
      prim_func = WithAttrs(std::move(prim_func),
                            {{tvm::attr::kTarget, host},
                             {tvm::attr::kCallingConv, Integer(CallingConv::kCPackedFunc)}});
      prim_func = WithoutAttr(std::move(prim_func), "vta.route_to_runtime");
    }
    mod->Update(global_var, std::move(prim_func));
  }
  return tir::transform::FlattenBuffer()(std::move(mod));
}

IRModule InjectConfigChecks(IRModule mod) {
  mod = mod->ShallowCopy();
  for (const auto& [global_var, base_func] : mod->functions) {
    tir::PrimFunc prim_func = Downcast<tir::PrimFunc>(base_func);
    tir::Call check(DataType::Int(32), tir::builtin::call_extern(),
                    {tir::StringImm("VTACheckConfig"),
                     IntImm(DataType::Int(64),
                            static_cast<int64_t>(static_cast<uint64_t>(VTA_ABI_FINGERPRINT)))});
    tir::Stmt fail = tir::Evaluate(
        tir::Call(DataType::Int(32), tir::builtin::tvm_throw_last_error(), {}));
    tir::Stmt guarded_body = tir::SeqStmt(
        {tir::IfThenElse(check != IntImm(DataType::Int(32), 0), fail), prim_func->body});
    prim_func.CopyOnWrite()->body = std::move(guarded_body);
    mod->Update(global_var, std::move(prim_func));
  }
  return mod;
}

}  // namespace

runtime::Module TIRToRuntime(IRModule mod, Target target) {
  ValidateModule(mod, target);
  Target host = target->GetHost().value();

  IRModule lowered = ForceFlattenExternalBuffers(std::move(mod));
  lowered = tir::transform::MakePackedAPI()(std::move(lowered));
  lowered = transform::Sequential({tir::transform::BindTarget(host),
                                   tir::transform::LowerTVMBuiltin(),
                                   tir::transform::LowerCustomDatatypes(),
                                   tir::transform::LowerIntrin(),
                                   tir::transform::LowerDeviceStorageAccessInfo(),
                                   tir::transform::CombineContextCall()})(std::move(lowered));
  lowered = InjectConfigChecks(std::move(lowered));
  return codegen::Build(std::move(lowered), host);
}

}  // namespace vta
}  // namespace tvm
