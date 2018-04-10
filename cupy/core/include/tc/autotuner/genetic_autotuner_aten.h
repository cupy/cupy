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

#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/genetic_autotuner.h"
#include "tc/autotuner/genetic_tuning_harness.h"
#include "tc/autotuner/utils/utils.h"
#include "tc/lang/parser.h"

namespace tc {
namespace autotune {

class GeneticAutotunerATen {
 public:
  GeneticAutotunerATen(const std::string tc);

  std::vector<MappingOptions> load(
      const std::string& cacheFileName,
      const std::string& tcName,
      const std::vector<at::Tensor> inputs,
      const size_t numCandidates);

  llvm::Optional<MappingOptions> tune(
      const std::string& cacheFileName,
      const std::string& tcName,
      const std::vector<at::Tensor>& inputs,
      MappingOptions baseMapping,
      std::vector<MappingOptions> startingPoints,
      const TuningParameterFixer& fixedParams = {});

 private:
  std::string tc_;
  std::unique_ptr<detail::GeneticAutotuner> geneticAutotuner_;
};

} // namespace autotune
} // namespace tc
