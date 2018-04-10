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
#include <iostream>
#include <memory>
#include <vector>

#include "tc/autotuner/genetic_tuning_harness.h"
#include "tc/autotuner/utils/utils.h"
#include "tc/lang/parser.h"

#include <llvm/ADT/Optional.h>

namespace tc {
namespace autotune {
namespace detail {

using Duration = std::chrono::high_resolution_clock::duration;

class GeneticAutotuner {
 public:
  explicit GeneticAutotuner(const std::string& tc);

  void storeCaches(const std::string& filename);

  std::vector<MappingOptions> load(
      const std::string& cacheFileName,
      const std::string& tcName,
      const std::vector<const DLTensor*>& inputs,
      const size_t numCandidates);

  llvm::Optional<MappingOptions> tune(
      const std::string& cacheFileName,
      const std::string& tcName,
      const std::unordered_map<size_t, std::vector<const DLTensor*>>& inputs,
      std::unordered_map<size_t, std::vector<DLTensor*>>& outputs,
      MappingOptions baseMapping,
      std::vector<MappingOptions> startingPoints,
      const TuningParameterFixer& fixedParams);

 private:
  std::string tc_;
  std::map<std::string, lang::TreeRef> tcNameMap_;
};

} // namespace detail
} // namespace autotune
} // namespace tc
