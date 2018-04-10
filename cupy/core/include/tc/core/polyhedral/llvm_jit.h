/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/Target/TargetMachine.h"

namespace tc {

namespace polyhedral {
class Scop;
}

class Jit {
 private:
  std::unique_ptr<llvm::TargetMachine> TM_;
  const llvm::DataLayout DL_;
  llvm::orc::RTDyldObjectLinkingLayer objectLayer_;
  llvm::orc::IRCompileLayer<decltype(objectLayer_), llvm::orc::SimpleCompiler>
      compileLayer_;

 public:
  Jit();

  void codegenScop(
      const std::string& specializedName,
      const polyhedral::Scop& scop);

  using ModuleHandle = decltype(compileLayer_)::ModuleHandleT;
  ModuleHandle addModule(std::unique_ptr<llvm::Module> M);
  void removeModule(ModuleHandle H);

  llvm::JITSymbol findSymbol(const std::string name);
  llvm::JITTargetAddress getSymbolAddress(const std::string name);

  llvm::TargetMachine& getTargetMachine();
};

} // namespace tc
