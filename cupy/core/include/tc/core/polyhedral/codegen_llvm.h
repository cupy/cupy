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

#include <memory>
#include <string>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

#include "Halide/Halide.h"

namespace tc {
namespace polyhedral {
struct Scop;

std::unique_ptr<llvm::Module> emitLLVMKernel(
    const std::string& specializedName,
    const Scop& scop,
    const llvm::DataLayout& dataLayout);

// TODO: I want to do something like the following, but compilation was unhappy
//  using initialize_llvm = Halide::Internal::CodeGen_LLVM::initialize_llvm;
static inline void initialize_llvm() {
  Halide::Internal::CodeGen_LLVM::initialize_llvm();
}

} // namespace polyhedral
} // namespace tc
