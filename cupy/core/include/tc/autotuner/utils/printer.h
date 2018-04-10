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
#include <iostream>
#include <mutex>
#include <thread>

#include <glog/logging.h>

#include "tc/core/rtc.h" // for Duration

namespace tc {
namespace autotune {

/**
 * Helper class to pretty print autotuning progress
 */
class Printer {
 public:
  Printer(
      size_t generation,
      size_t total,
      const std::atomic_size_t& currentCompilationJob,
      const std::atomic_size_t& numEvaluations);
  ~Printer();

  void record(Duration runtime);
  void stop();

  void printAll();

 private:
  void printLoop();

  size_t generation_;
  std::vector<Duration> runtimes_;
  mutable std::mutex runtimesMtx_;

  std::atomic_bool stopPrinting_{false};
  std::thread printerThread_;

  const size_t total_;
  const std::atomic_size_t& currentCompilationJob_;
  const std::atomic_size_t& numEvaluations_;
};

} // namespace autotune
} // namespace tc
