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

namespace tc {
namespace autotune {

template <typename T>
void ConcurrentQueue<T>::enqueue(T t) {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    queue_.push(t);
  }
  cv_.notify_one();
}

template <typename T>
T ConcurrentQueue<T>::dequeueWaitFor(std::chrono::steady_clock::duration d) {
  std::unique_lock<std::mutex> lock(mtx_);
  auto hasElements =
      cv_.wait_for(lock, d, [&]() { return not queue_.empty(); });
  if (not hasElements) {
    return nullptr;
  }
  auto t = queue_.front();
  queue_.pop();
  return t;
}

template <typename T>
bool ConcurrentQueue<T>::empty() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return queue_.empty();
}

template <typename T>
size_t ConcurrentQueue<T>::size() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return queue_.size();
}

} // namespace autotune
} // namespace tc
