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

#include "tc/core/utils/function_traits.h"

#include <functional>
#include <vector>

namespace tc {
namespace polyhedral {
namespace functional {

template <typename I>
void App(std::function<void(I)> fun, const std::vector<I>& vec) {
  for (auto& i : vec) {
    fun(i);
  }
}

template <typename I>
void App(std::function<void(I)> fun, const std::vector<I>&& vec) {
  App(fun, vec);
}

template <typename I>
void App(std::function<void(I, size_t)> fun, const std::vector<I>& vec) {
  size_t pos = 0;
  for (auto& i : vec) {
    fun(i, pos++);
  }
}

template <typename I>
void App(std::function<void(I, size_t)> fun, const std::vector<I>&& vec) {
  App(fun, vec);
}

template <typename R, typename I>
std::vector<R> Map(std::function<R(I)> fun, const std::vector<I>& vec) {
  std::vector<R> res;
  res.reserve(vec.size());
  for (auto& i : vec) {
    res.push_back(fun(i));
  }
  return res;
}

template <typename R, typename I>
std::vector<R> Map(std::function<R(I)> fun, std::vector<I>&& vec) {
  return Map<R, I>(fun, vec);
}

template <typename I>
I Reduce(
    std::function<I(const I&, const I&)> red,
    I initVal,
    const std::vector<I>& vec) {
  I res = initVal;
  for (int i = 0; i < vec.size(); ++i) {
    res = red(res, vec.at(i));
  }
  return res;
}

template <typename I>
I Reduce(std::function<I(const I&, const I&)> red, const std::vector<I>& vec) {
  const std::vector<I> v(vec.begin() + 1, vec.end());
  return Reduce(red, vec.at(0), v);
}

template <typename I>
I Reduce(std::function<I(I&&, I&&)> red, I&& initVal, std::vector<I>&& vec) {
  I res = std::move(initVal);
  for (int i = 0; i < vec.size(); ++i) {
    res = red(std::move(res), std::move(vec.at(i)));
  }
  return res;
}

template <typename I>
I Reduce(std::function<I(I&&, I&&)> red, std::vector<I>&& vec) {
  I res = std::move(vec.at(0));
  for (int i = 1; i < vec.size(); ++i) {
    res = red(std::move(res), std::move(vec.at(i)));
  }
  return res;
}

/*
 * Return a copy of the vector that contains only those elements for which
 * function "f" returns true.
 *
 * The first argument must be a function-like entity (function pointer,
 * functor) with signature <bool(T)>.
 *
 * Template argument deduction takes care of vector<const T> cases
 * automatically.  Note that the signature of "f" must use the same type, that
 * is "const T".
 */
template <typename Func, typename T>
std::vector<T> Filter(Func f, const std::vector<T>& input) {
  static_assert(
      std::is_same<typename function_traits<Func>::result_type, bool>::value,
      "Filtering function must return bool");
  static_assert(
      function_traits<Func>::n_args == 1,
      "Filtering function must take one argument");
  static_assert(
      std::is_same<typename function_traits<Func>::template arg<0>::type, T>::
          value,
      "The argument of the filtering function must have the same type "
      "as the element type of the collection being filtered");

  std::vector<T> res;
  res.reserve(input.size());
  for (const auto& i : input) {
    if (f(i)) {
      res.push_back(i);
    }
  }
  res.shrink_to_fit();
  return res;
}

template <typename R, typename I>
R MapReduce(std::function<R(R, I, bool)> fun, const std::vector<I>& vec) {
  R res = fun(R(), vec.at(0), true);
  for (int i = 1; i < vec.size(); ++i) {
    res = fun(res, vec.at(i), false);
  }
  return res;
}

} // namespace functional
} // namespace polyhedral
} // namespace tc
