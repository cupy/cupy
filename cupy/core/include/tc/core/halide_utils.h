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

#include <chrono>
#include <string>
#include <unordered_set>
#include <vector>

#include "tc/core/tc2halide.h"
#include "tc/core/utils/cuda_info.h"
#include "tc/core/utils/dlpack.h"

namespace tc {

/// Given the result of translating TC language to Halide as components and the
/// (metadata of) input tensors with specific shapes, compute a map between TC
/// parametric tensor sizes, represented as strings, and their numerical values
/// with given input sizes.
std::map<std::string, int> computeParamValueMap(
    const tc2halide::HalideComponents& components,
    const std::vector<const DLTensor*>& inputsDLT);

/// Infer the numerical sizes of the output tensors in the TC definition
/// translated into Halide using the provided map between symbolic parameter
/// names and their values ("pvm").
/// @return metadata of the output tensors, with ownership transfer via
/// unique_ptr, data pointers of the underlying DLTensors are null/
std::vector<dlutils::DLTensorUPtr> inferOutputTensorInfo(
    const tc2halide::HalideComponents& halide,
    const std::vector<const DLTensor*>& inputsDLT);

/// Just generates a C function body from a Halide stmt. Exposed for testing.
std::string halideCodegenC(const Halide::Internal::Stmt& s);

} // namespace tc
