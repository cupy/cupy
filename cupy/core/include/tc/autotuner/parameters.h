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
#include <functional>
#include <memory>
#include <vector>

#include "tc/core/mapping_options.h"
#include "tc/core/utils/memory.h"

#include <llvm/ADT/Optional.h>

namespace tc {
namespace autotune {

class ParameterView;

class BoolParameter {
 public:
  BoolParameter(const std::string& name);
  BoolParameter() = default;
  BoolParameter(const BoolParameter&);
  BoolParameter& operator=(const BoolParameter&);

  size_t numberOptions() const;
  void selectOption(size_t idx);
  void selectValue(bool val);
  void fixValue(bool val);

  void apply(const std::function<void(ParameterView&)>& f);

  bool value() const;

  std::string name;

 private:
  friend class ParameterView;
  bool value_;
  llvm::Optional<bool> fixedValue_;
};

class RangeParameter {
 public:
  RangeParameter(std::vector<size_t> values, const std::string& name);
  RangeParameter(const RangeParameter&);
  RangeParameter() = default;
  RangeParameter& operator=(const RangeParameter&);

  size_t numberOptions() const;
  void selectOption(size_t idx);
  void selectFromValue(size_t v);
  void fixValue(size_t v);
  size_t value() const;

  void apply(const std::function<void(ParameterView&)>& f);

  std::string name;

 private:
  friend class ParameterView;
  size_t selected_ = 0;
  std::vector<size_t> values_;
  llvm::Optional<size_t> fixedValue_;
};

class ParameterView {
 public:
  ParameterView(BoolParameter&);
  ParameterView(RangeParameter&);

  size_t numberOptions() const;
  void selectOption(size_t idx);
  void overwrite(const ParameterView&);
  bool isForced() const;

 private:
  RangeParameter* rangePtr;
  BoolParameter* boolPtr;
};

class SchedulerOptionsParameters {
 public:
  SchedulerOptionsParameters();

  void fromMappingOptions(const SchedulerOptionsView& options);
  void applyToMappingOptions(SchedulerOptionsView& options) const;

  void apply(const std::function<void(ParameterView&)>& f);
  std::vector<ParameterView> collectParameters();

  RangeParameter fusionStrategy;
  // BoolParameter allowSkewing;
  // BoolParameter positiveOrthant;
};

class MultiRangeParams {
 protected:
  void setRange(
      size_t minDims,
      size_t maxDims,
      const std::string& name,
      std::vector<size_t>& values,
      const std::string& dimBaseName);

 public:
  void apply(const std::function<void(ParameterView&)>& f);
  std::vector<ParameterView> collectParameters();
  RangeParameter numberDims;
  std::vector<RangeParameter> dims;
};

class TilingParameters : public MultiRangeParams {
 public:
  void setRange(size_t maxDims, std::vector<size_t>& values);
  void fromMappingOptions(const TilingView& options);
  void applyToMappingOptions(TilingView& options) const;

  using MultiRangeParams::apply;
  using MultiRangeParams::collectParameters;
};

class CudaDimParameters : public MultiRangeParams {
 public:
  void setRange(
      std::vector<size_t>& values,
      const std::string& dimensionBaseName);
  void fromMappingOptions(const CudaDimView& options);
  void applyToMappingOptions(CudaDimView& options) const;

  using MultiRangeParams::apply;
  using MultiRangeParams::collectParameters;
};

class TuningParameterFixer;

class TuningConfiguration {
 public:
  void applyToParameters(const std::function<void(ParameterView&)>& f);
  std::vector<ParameterView> collectParameters();

  TuningConfiguration();
  TuningConfiguration(const MappingOptions&);
  TuningConfiguration(const TuningConfiguration&) = default;
  TuningConfiguration& operator=(const TuningConfiguration&) = default;

  void fromMappingOptions(const MappingOptions& options);
  void applyToMappingOptions(MappingOptions& options) const;

  void addValidator(std::function<bool(const TuningConfiguration&)> v);
  bool isValid() const;

  void fixParameters(const TuningParameterFixer& fixedParams);

  friend std::ostream& operator<<(
      std::ostream& os,
      const TuningConfiguration& conf);

  SchedulerOptionsParameters outerScheduleOptions;
  SchedulerOptionsParameters intraTileScheduleOptions;
  BoolParameter fixParametersBeforeScheduling;
  TilingParameters tilingParams;
  CudaDimParameters blockParams;
  CudaDimParameters gridParams;
  RangeParameter unrollFactor;
  BoolParameter tileImperfectlyNested;
  BoolParameter useSharedMemory;
  BoolParameter usePrivateMemory;
  BoolParameter unrollCopyShared;
  BoolParameter matchLibraryCalls;

 private:
  std::vector<std::function<bool(const TuningConfiguration&)>> validators_;
};

class TuningParameterFixer {
 public:
  TuningParameterFixer& fixOuterScheduleFusionStrategy(
      const FusionStrategy& fs);
  TuningParameterFixer& fixIntraTileScheduleFusionStrategy(
      const FusionStrategy& fs);
  TuningParameterFixer& fixFixParametersBeforeScheduling(bool val);
  TuningParameterFixer& fixUnrollFactor(size_t val);
  TuningParameterFixer& fixTilingParameters(std::vector<size_t> vals);
  TuningParameterFixer& fixBlockParameters(std::vector<size_t> vals);
  TuningParameterFixer& fixGridParameters(std::vector<size_t> vals);
  TuningParameterFixer& fixTileImperfectlyNested(bool val);
  TuningParameterFixer& fixUseSharedMemory(bool val);
  TuningParameterFixer& fixUsePrivateMemory(bool val);
  TuningParameterFixer& fixUnrollCopyShared(bool val);
  TuningParameterFixer& fixMatchLibraryCalls(bool val);

 private:
  llvm::Optional<FusionStrategy> outerScheduleFusionStrategy;
  llvm::Optional<FusionStrategy> intraTileScheduleFusionStrategy;
  llvm::Optional<bool> fixParametersBeforeScheduling;
  llvm::Optional<size_t> unrollFactor;
  llvm::Optional<std::vector<size_t>> tilingParameters;
  llvm::Optional<std::vector<size_t>> blockParameters;
  llvm::Optional<std::vector<size_t>> gridParameters;
  llvm::Optional<bool> tileImperfectlyNested;
  llvm::Optional<bool> useSharedMemory;
  llvm::Optional<bool> usePrivateMemory;
  llvm::Optional<bool> unrollCopyShared;
  llvm::Optional<bool> matchLibraryCalls;

  friend class TuningConfiguration;
};

using TimePoint = std::chrono::high_resolution_clock::time_point;
using Duration = std::chrono::high_resolution_clock::duration;

class CandidateConfiguration {
 public:
  CandidateConfiguration(
      const TuningConfiguration& config,
      Duration d = Duration::zero(),
      bool invalid = false)
      : configuration(config),
        runtime(d),
        invalid(invalid),
        optionalCompilationHandle(nullptr) {}

  CandidateConfiguration(const CandidateConfiguration& candidate)
      : configuration(candidate.configuration),
        runtime(candidate.runtime),
        invalid(candidate.invalid),
        optionalCompilationHandle(
            candidate.optionalCompilationHandle
                ? std::unique_ptr<size_t>(
                      new size_t(*candidate.optionalCompilationHandle))
                : nullptr) {}

  CandidateConfiguration& operator=(const CandidateConfiguration& candidate) {
    CandidateConfiguration tmp(candidate);
    std::swap(tmp, *this);
    return *this;
  }

  friend std::ostream& operator<<(
      std::ostream& os,
      const CandidateConfiguration& config) {
    return os << config.configuration;
  }

  TuningConfiguration configuration;
  Duration runtime;
  bool invalid;
  std::unique_ptr<size_t> optionalCompilationHandle;
};

} // namespace autotune
} // namespace tc
