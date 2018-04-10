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

#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include <string>

#include <glog/logging.h>
#include <version.h>

namespace tc {

template <typename CC>
void Cache<CC>::enableCache() {
  CC::getGlobalSharedCache() = std::make_shared<CC>();
}

template <typename CC>
void Cache<CC>::disableCache() {
  CC::getGlobalSharedCache() = nullptr;
}

template <typename CC>
std::shared_ptr<CC> Cache<CC>::getCache() {
  if (not cacheEnabled()) {
    throw std::runtime_error(
        "EnableCache or LoadCacheFromProtobuf must be called before using the cache.");
  }
  return CC::getGlobalSharedCache();
}

template <typename CC>
void Cache<CC>::dumpCacheToProtobuf(const std::string& filename) {
  std::fstream serialized(
      filename, std::ios::binary | std::ios::trunc | std::ios::out);
  if (!serialized) {
    LOG(ERROR) << "Failed to open the output stream for dumping protobuf: "
               << filename;
  } else {
    getCache()->toProtobuf().SerializePartialToOstream(&serialized);
  }
}

template <typename CC>
void Cache<CC>::loadCacheFromProtobuf(const std::string& filename) {
  typename CC::Protobuf buf;
  struct stat buffer = {0};
  if (stat(filename.c_str(), &buffer) == 0) {
    std::ifstream serialized(filename, std::ios::binary);
    buf.ParseFromIstream(&serialized);
  }
  loadCacheFromProtobuf(buf);
}

template <typename CC>
template <typename Protobuf>
void Cache<CC>::loadCacheFromProtobuf(const Protobuf& buf) {
  static_assert(
      std::is_same<Protobuf, typename CC::Protobuf>::value,
      "LoadCacheFromProtobuf called with invalide protobuf type.");
  CC::getGlobalSharedCache() = std::make_shared<CC>(buf);
}

template <typename CC>
bool Cache<CC>::cacheEnabled() {
  return CC::getGlobalSharedCache() != nullptr;
}

template <typename CC>
size_t Cache<CC>::size() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return static_cast<const CC*>(this)->entries_.size();
}

template <typename CC>
void Cache<CC>::clear() {
  std::lock_guard<std::mutex> lock(mtx_);
  numberAttemptedRetrievals = numberSuccessfulRetrievals = numberCacheAttemps =
      0;
  static_cast<CC*>(this)->entries_.clear();
}

template <typename C, typename InputTy> // deduces whether C is const or
// non-const
auto CudaCache::searchKernelImpl(
    C& c,
    const std::string& id,
    const MappingOptions& options,
    const std::vector<InputTy>& inputs,
    const std::vector<InputTy>& outputs)
    -> decltype(c.searchKernel(id, options, inputs, outputs)) {
  auto gpuStr = CudaGPUInfo::GPUInfo().GetCudaDeviceStr();
  auto it = std::find_if(
      c.entries_.begin(), c.entries_.end(), [&](const CachedEntry& c) {
        using tc::operator==;
        return id == c.key.id && options == c.key.mappingOptions &&
            inputs == c.key.inputs && outputs == c.key.outputs &&
            gpuStr == c.key.deviceStr;
      });
  if (it != c.entries_.end()) {
    if (it->key.gitVersion != tc::git_version) {
      std::cerr << "Proto version doesn't match. TC git version is: "
                << tc::git_version
                << " and Proto version is: " << it->key.gitVersion
                << " .This proto might be incompatible"
                << " with your TC binary and can break. Please autotune"
                << " against the correct TC version." << std::endl;
    }
    return &*it;
  }
  return nullptr;
}

// deduces whether C is const or non-const
template <typename C>
auto OptionsCache::searchKernelImpl(
    C& c,
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs)
    -> decltype(c.searchKernel(id, inputs, outputs)) {
  auto gpuStr = CudaGPUInfo::GPUInfo().GetCudaDeviceStr();
  auto it = std::find_if(
      c.entries_.begin(), c.entries_.end(), [&](const CachedEntry& c) {
        using tc::operator==;
        return id == c.key.id && inputs == c.key.inputs &&
            outputs == c.key.outputs && gpuStr == c.key.deviceStr;
      });
  if (it != c.entries_.end()) {
    if (it->key.gitVersion != tc::git_version) {
      std::cerr << "Proto version doesn't match. TC git version is: "
                << tc::git_version
                << " and Proto version is: " << it->key.gitVersion
                << " .This proto might be incompatible"
                << " with your TC binary and can break. Please autotune"
                << " against the correct TC version." << std::endl;
      ;
    }
    return &*it;
  }
  return nullptr;
}

// deduces whether C is const or non-const
template <typename C, typename TensorTy>
auto ManualCudaCache::searchKernelImpl(
    C& c,
    const std::string& id,
    const std::vector<TensorTy>& inputs,
    const std::vector<TensorTy>& outputs)
    -> decltype(c.searchKernel(id, inputs, outputs)) {
  auto gpuStr = CudaGPUInfo::GPUInfo().GetCudaDeviceStr();
  auto it = std::find_if(
      c.entries_.begin(), c.entries_.end(), [&](const CachedEntry& c) {
        using tc::operator==;
        return id == c.key.id && inputs == c.key.inputs &&
            outputs == c.key.outputs && gpuStr == c.key.deviceStr;
      });
  if (it != c.entries_.end()) {
    std::cout << "RETURNING IT: " << it->key.gitVersion << std::endl;
    if (it->key.gitVersion != tc::git_version) {
      std::cerr << "Proto version doesn't match. TC git version is: "
                << tc::git_version
                << " and Proto version is: " << it->key.gitVersion
                << " .This proto might be incompatible"
                << " with your TC binary and can break. Please autotune"
                << " against the correct TC version." << std::endl;
      ;
    }
    return &*it;
  }
  return nullptr;
}

} // namespace tc
