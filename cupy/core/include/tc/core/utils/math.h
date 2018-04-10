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

#include <exception>

namespace tc {

template <typename V>
typename V::value_type median(V v) {
  if (v.size() == 0) {
    throw std::out_of_range("median not defined for empty containers");
  }

  auto n = v.size();
  std::nth_element(v.begin(), v.begin() + n / 2, v.end());
  if (n % 2 == 1) {
    return v.at(n / 2);
  }
  auto rightElement = v.at(n / 2);
  std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
  return (v.at(n / 2 - 1) + rightElement) / 2;
}

} // namespace tc
