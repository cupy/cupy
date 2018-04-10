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

#include <string>
#include <vector>

namespace tc {

//
// This functionality in this type of class has been rewritten over and over
// again. Here we just provide a static singleton and basic properties.
// Consider lifting stuff up from fbcuda rather than reinventing the wheel
//
class CudaGPUInfo {
  CudaGPUInfo(const std::vector<std::string>& gpuNames) : gpuNames_(gpuNames) {}

 public:
  static CudaGPUInfo& GPUInfo();

  // These functions require init to have been run, they are thus members of
  // the singleton object and not static functions.
  int NumberGPUs() const;
  int CurrentGPUId() const;
  void SynchronizeCurrentGPU() const;
  std::string GetGPUName(int id = -1) const;
  std::string GetCudaDeviceStr() const;

  std::vector<std::string> gpuNames_;
};

} // namespace tc
