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

#include <type_traits>
#include <vector>

namespace tc {
namespace {

/// Helper functions to construct vectors from a list of identically-typed
/// function arguments.  Depending on the argument type being lvalue or rvalue
/// (reference), the vector elements are copy- or move-inserted.
template <typename T>
void inplaceVectorFromArgs(std::vector<T>&) {
  // recursion stopper
}

template <typename T, typename Arg, typename... Args>
void inplaceVectorFromArgs(std::vector<T>& v, Arg arg, Args... args) {
  static_assert(
      std::is_same<T, typename std::remove_reference<Arg>::type>::value,
      "expected identical types");
  v.push_back(std::forward<Arg>(arg));
  inplaceVectorFromArgs(v, std::forward<Args>(args)...);
}
} // namespace

/// Empty list of arguments gives an empty vector. The function call has to be
/// fully specialized.
template <typename Arg>
std::vector<Arg> vectorFromArgs() {
  std::vector<Arg> v;
  return v;
}

/// The type of the vector is deduced from the first argument type.
template <typename Arg, typename... Args>
std::vector<typename std::remove_reference<Arg>::type> vectorFromArgs(
    Arg arg,
    Args... args) {
  std::vector<typename std::remove_reference<Arg>::type> v;
  inplaceVectorFromArgs(v, std::forward<Arg>(arg), std::forward<Args>(args)...);
  return v;
}

template <typename Arg, typename... Args>
std::vector<typename std::remove_reference<Arg>::type> vectorFromCastedArgs(
    Args... args) {
  std::vector<typename std::remove_reference<Arg>::type> v;
  inplaceVectorFromArgs(v, static_cast<Arg>(args)...);
  return v;
}

template <template <class> class ConditionType, typename T, typename... Args>
struct TemplArgsAll {
  const static bool value =
      ConditionType<T>::value && TemplArgsAll<ConditionType, Args...>::value;
};

template <template <class> class ConditionType, typename T>
struct TemplArgsAll<ConditionType, T> {
  const static bool value = ConditionType<T>::value;
};

template <template <class> class ConditionType>
struct TemplArgsAll<ConditionType, void> {
  const static bool value = true;
};

} // namespace tc
