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
#include <algorithm>
#include <iterator>
#include <type_traits>
#include <vector>

namespace tc {
namespace autotune {

namespace detail {
template <typename Vector>
void mergeVectorsImpl(Vector&) {}

template <typename Vector, typename... Vectors>
void mergeVectorsImpl(Vector& sink, Vector&& v, Vectors&&... vs) {
  mergeVectorsImpl(sink, std::forward<Vectors>(vs)...);

  if (std::is_rvalue_reference<decltype(v)>::value) {
    sink.reserve(sink.size() + v.size());
    std::move(v.begin(), v.end(), std::back_inserter(sink));
  } else {
    sink.insert(sink.end(), v.begin(), v.end());
  }
}

} // namespace detail

template <typename Vector, typename... Vectors>
Vector mergeVectors(Vector&& v, Vectors&&... vs) {
  Vector merged;
  detail::mergeVectorsImpl(
      merged, std::forward<Vector>(v), std::forward<Vectors>(vs)...);
  std::sort(merged.begin(), merged.end());
  merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
  return merged;
}

} // namespace autotune
} // namespace tc
