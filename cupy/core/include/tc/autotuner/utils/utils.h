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
#include <vector>

#include <ATen/ATen.h>

#include "tc/core/cuda.h"
#include "tc/core/mapping_options.h"
#include "tc/core/utils/dlpack.h"

#include <llvm/ADT/Optional.h>

namespace tc {
namespace autotune {

struct OptionsWithMedianTime {
  MappingOptions options;
  Duration medianRuntime;
};

/// Returns all the powers of 2 up to the first one that is larger than val
/// and the result of ceil(val/pow2) for each of those powers of 2 (except for
/// the larger one)
std::vector<std::size_t> powers2andCeilDivisors(std::size_t val);

template <typename Vector, typename... Vectors>
Vector mergeVectors(Vector&& v, Vectors&&... vs);

std::vector<OptionsWithMedianTime> getOptionsAndMedianRuntimes(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs);

std::vector<MappingOptions> restoreCandidates(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs);

llvm::Optional<MappingOptions> getBestOptions(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs);

} // namespace autotune
} // namespace tc

#include "tc/autotuner/utils/utils-inl.h"
