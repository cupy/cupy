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

#include <Halide/Halide.h>

#include "tc/external/isl.h"
#include "tc/lang/tree.h"
#include "tc/lang/tree_views.h"

namespace tc2halide {

// We lower TC source into a Halide imperative statement, plus a list
// of the input and output tensors. We do not explicitly enumerate the
// scalar params.
struct HalideComponents {
  lang::TreeRef
      def; // post-semantic analaysis tree, used for later error reporting
  Halide::Internal::Stmt stmt;
  std::vector<Halide::ImageParam> inputs;
  std::map<std::string, Halide::Internal::Parameter> params;
  std::vector<Halide::OutputImageParam> outputs;
  lang::Def getDef() const {
    return lang::Def(def); // Def is not default constructable, so we don't
                           // put it in the struct directly
  }
};

// For TC reductions, the right-hand-sides of the corresponding
// Provide nodes are tagged with intrinsics with the following names.
Halide::Internal::Call::ConstString kReductionInit = "ReductionInit";
Halide::Internal::Call::ConstString kReductionUpdate = "ReductionUpdate";

// Translate a TC parse tree into equivalent Halide imperative IR with
// a naive schedule.
HalideComponents translate(
    isl::ctx ctx,
    const lang::TreeRef& treeRef,
    bool throwWarnings = false);

// Translate TC source into equivalent Halide imperative IR with a
// naive schedule.
HalideComponents
translate(isl::ctx ctx, const std::string& tc, bool throwWarnings = false);

} // namespace tc2halide
