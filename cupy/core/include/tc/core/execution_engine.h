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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include "tc/core/mapping_options.h"
#include "tc/core/tc_executor.h"

namespace tc {

/// The goal for this new shiny API is to provide a different pathway for being
/// able to execute the kernels for multiple TC i.e. given the language which
/// can have multiple TCs, people should be able to run things by just calling
/// out the run function with the name of function and the inputs to run on.
class ExecutionEngine {
 public:
  struct ExecutorInfo {
    ExecutorInfo(
        std::string id,
        std::vector<const DLTensor*> inputsInfo,
        std::unique_ptr<MappingOptions> options,
        lang::TreeRef tc,
        size_t handle)
        : identifier(id),
          inputsInfo(dlutils::makeDLTensorVector(inputsInfo)),
          options(std::move(options)),
          exec(tc, inputsInfo),
          objectLocalHandle(handle) {}

    void clear() {
      exec.clearRTC();
    }

    std::string identifier;
    std::vector<dlutils::DLTensorUPtr> inputsInfo;
    std::unique_ptr<MappingOptions> options;
    TcExecutor exec;
    /// When run is called this is used to find the most recently compiled
    /// version.
    size_t objectLocalHandle;
  };

  ExecutionEngine() = default;

  /// Create the ExecutionEngine::tcNameMap_ using the language passed
  /// to it - should support many TC.
  void define(const std::string& language);

  /// Create the ExecutionEngine::tcNameMap_ from the parsed TC
  /// string - supports many TC.
  void define(const std::vector<lang::TreeRef>& treeRefs);

  void addTC(const std::string& tc);

  /// Get the output Tensor info that can be used by the calling framework to
  /// allocate storage for the output.
  std::vector<const DLTensor*> inferOutputTensorInfo(
      const std::string& name,
      const std::vector<const DLTensor*>& inTensorPtrs);

  lang::TreeRef treeForFunction(const std::string& name) {
    return tcNameMap_.at(name);
  }

  // TODO: Pass autotuning info (none by default, otherwise some struct with
  //       maxtime and other things)

  /// Returns a handle for the compiled kernel
  size_t compile(
      const std::string& name,
      const std::vector<const DLTensor*>& inputs,
      const MappingOptions& options);

  // TODO: sanity check on name and input / output sizes.
  /// Run a TC specified by its name on the given tensor inputs and fill the
  /// outputs with the result.
  /// The TC is looked up by its handle.
  /// If profile is set, the kernel runtime is returned.
  ///
  /// The pruning function returns true if the run should not proceed (e.g. if
  /// there are too few threads mapped that would likely result in catastrophic
  /// performance). In this case, return Duration::max().
  Duration run(
      size_t handle,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<DLTensor*>& outputs,
      bool profile = false,
      std::function<bool(const ExecutorInfo*)> pruningFunction =
          [](const ExecutorInfo*) { return false; });

  /// This is the "low-latency" mode in which we just propagate raw pointers to
  /// data in GPU address space.
  /// No tensor-related information can be checked so it is the user's
  /// responsibility to ensure that shapes and strides match. If the user
  /// doesn't then segfault will likely occur.
  void uncheckedRun(
      size_t handle,
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs);

  void clear(size_t handle);

 private:
  size_t getHandle(
      const std::string& name,
      const std::vector<const DLTensor*>& inputsInfo,
      const MappingOptions& options);
  std::unique_ptr<ExecutorInfo> makeExecutorInfo(
      const std::string& name,
      const std::vector<const DLTensor*>& inputsInfo,
      const MappingOptions& options);
  size_t emplaceExecutor(std::unique_ptr<ExecutorInfo> p);

  /// For thread-safety perform all cheap operations under lock
  std::mutex executorInfoMutex;

  // XXX:if ExecutorInfo is moved/copied (even when the vector's underlying
  // storage is extended) something inside isl segfaults,  unique_ptr is used as
  // a workaround
  std::vector<std::unique_ptr<ExecutorInfo>> executors_;
  std::map<std::string, lang::TreeRef> tcNameMap_;

  size_t uidCounter = 0;
};

} // namespace tc
