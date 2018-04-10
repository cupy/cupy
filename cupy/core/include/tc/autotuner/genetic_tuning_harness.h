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

#include <atomic>
#include <csignal>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/genetic_search.h"
#include "tc/autotuner/parameters.h"
#include "tc/autotuner/utils/printer.h"
#include "tc/lang/parser.h"

namespace tc {
namespace autotune {
namespace detail {

extern volatile std::sig_atomic_t signal_;
extern volatile std::sig_atomic_t killRequested_;

class GeneticTunerHarness {
 public:
  GeneticTunerHarness(
      size_t n,
      uint8_t crossoverRate,
      uint8_t mutationRate,
      size_t numberElites,
      lang::TreeRef tc,
      std::string kernelName,
      const std::unordered_map<size_t, std::vector<const DLTensor*>>& inputs,
      std::unordered_map<size_t, std::vector<DLTensor*>>& outputs,
      MappingOptions baseMapping,
      std::vector<MappingOptions> startingPoints,
      const TuningParameterFixer& fixedParams);
  void run(size_t numGenerations);

 private:
  void setupTuningParameters();

  /// Traverse one generation of candidates in parallel and evaluate their
  /// runtimes
  void runOneGeneration(size_t generation);

  /// Helper function to get a kernel into benchmark-able state
  bool warmupOrPrune(
      tc::ExecutionEngine& executionEngine,
      const std::vector<DLTensor*>& outputs,
      const std::vector<const DLTensor*>& inputs,
      size_t handle,
      size_t bestTimeSoFar);

  /// Helper function to delegate compiling on the cpu to different threads
  void doCompile(tc::ExecutionEngine& engine);
  /// Helper function to delegate running on the gpu to different threads
  void doGpuWork(size_t gpu, tc::ExecutionEngine& engine, Printer& printer);

  /// Make options from conf
  tc::MappingOptions makeOptions(const CandidateConfiguration& conf);
  TuningConfiguration makeTuningConfiguration(const MappingOptions& options);
  MappingOptions bestMappingOption() {
    std::lock_guard<std::mutex> lock(bestTimeMtx_);
    return bestMappingOptions_;
  }

 public:
  static constexpr int kReducedWarmupIterations = 2;
  static constexpr int kReducedBenchmarkIterations = 10;
  static constexpr int kEarlyPruneFactor = 5;

  const size_t kMaxPopulationSize;
  const uint8_t kCrossOverRate;
  const uint8_t kMutationRate;
  const size_t kNumberElites;

  TuningConfiguration configuration;

 private:
  std::mutex bestTimeMtx_;
  size_t bestTime_ = std::numeric_limits<size_t>::max();
  MappingOptions bestMappingOptions_;

  const lang::TreeRef kTc_;
  const std::string kKernelName_;
  std::unique_ptr<GeneticSearch> tuner_;
  std::atomic_size_t currentCompilationJob_;
  std::deque<std::atomic_bool> readyToEvaluate_;
  std::atomic_size_t numEvaluations_;
  const std::unordered_map<size_t, std::vector<const DLTensor*>> kInputs_;
  std::unordered_map<size_t, std::vector<DLTensor*>> outputs_;
  const MappingOptions kBaseMapping_;
  const std::vector<MappingOptions> kStartingPoints_;
};

std::vector<size_t> parseGpus();

} // namespace detail
} // namespace autotune
} // namespace tc
