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
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <driver_types.h> // cuda driver types

namespace tc {

extern std::mutex nvrtc_mutex;
using Duration = std::chrono::high_resolution_clock::duration;

//
// Basic interface to expose NVRTC JIT compilation and module
// loading/unloading + API kernel launches.
//
class CudaRTCFunction {
  CudaRTCFunction();

 public:
  ~CudaRTCFunction();

  static std::shared_ptr<CudaRTCFunction> Compile(
      const std::string& name,
      const std::string& source);

  // if profile is set it returns the kernel runtime
  Duration Launch(
      const std::array<size_t, 3>& grid,
      const std::array<size_t, 3>& block,
      unsigned int shared_mem,
      cudaStream_t stream,
      // by copy because we take an address to element when calling the kernel
      // TODO: check the overhead of double indirection on kernel calls, this
      // does not look ideal for low-latency
      std::vector<int> params,
      std::vector<void*> outputs,
      std::vector<const void*> inputs,
      bool profile = false) const;

  void clear();

 private:
  mutable std::unordered_map<size_t, CUmodule> perGpuModule_;
  mutable std::unordered_map<size_t, CUfunction> perGpuKernel_;
  std::string specializedName;
  std::vector<char> nvrtc_ptx;
  bool cleared_;
};

} // namespace tc
