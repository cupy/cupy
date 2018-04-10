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

#ifndef CUDA_HOME
#error "CUDA_HOME must be defined"
#endif // CUDA_HOME

#ifndef CUB_HOME
#error "CUB_HOME must be defined"
#endif // CUB_HOME

#include <sstream>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

#define TC_CUDA_DRIVERAPI_ENFORCE(condition)                            \
  do {                                                                  \
    CUresult result = condition;                                        \
    if (result != CUDA_SUCCESS) {                                       \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      std::stringstream ss;                                             \
      ss << "Error at: " << __FILE__ << ":" << __LINE__ << ": " << msg; \
      LOG(WARNING) << ss.str();                                         \
      throw std::runtime_error(ss.str().c_str());                       \
    }                                                                   \
  } while (0)

#define TC_NVRTC_CHECK(condition)                               \
  do {                                                          \
    nvrtcResult result = condition;                             \
    if (result != NVRTC_SUCCESS) {                              \
      std::stringstream ss;                                     \
      ss << "Error at: " << __FILE__ << ":" << __LINE__ << ": " \
         << nvrtcGetErrorString(result);                        \
      LOG(WARNING) << ss.str();                                 \
      throw std::runtime_error(ss.str().c_str());               \
    }                                                           \
  } while (0)

#define TC_CUDA_RUNTIMEAPI_ENFORCE(condition)                   \
  do {                                                          \
    cudaError_t result = condition;                             \
    if (result != cudaSuccess) {                                \
      std::stringstream ss;                                     \
      ss << "Error at: " << __FILE__ << ":" << __LINE__ << ": " \
         << cudaGetErrorString(result);                         \
      LOG(WARNING) << ss.str();                                 \
      throw std::runtime_error(ss.str().c_str());               \
    }                                                           \
  } while (0)

namespace tc {

struct WithDevice {
  WithDevice(size_t g) : newGpu(g) {
    int dev;
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaGetDevice(&dev));
    oldGpu = dev;
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaSetDevice(newGpu));
  }
  ~WithDevice() noexcept(false) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaSetDevice(oldGpu));
  }
  size_t oldGpu;
  size_t newGpu;
};

// Query the active device about the avaialble memory size.
size_t querySharedMemorySize();

} // namespace tc
