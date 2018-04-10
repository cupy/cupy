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

#include <functional>
#include <type_traits>

// Type traits class for functions.
// Allows to query result and argument types of free and member functions as
// well as of operator() of functors.

namespace tc {

template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

#define FUNCTION_TRAITS_STRUCT_BODY                                         \
  typedef R result_type;                                                    \
                                                                            \
  template <size_t i>                                                       \
  struct arg {                                                              \
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type; \
  };                                                                        \
                                                                            \
  typedef std::tuple<Args...> packed_args_type;                             \
  typedef R (*c_function_type)(Args...);                                    \
  constexpr static size_t n_args = sizeof...(Args)

template <typename ClassType, typename R, typename... Args>
struct function_traits<R (ClassType::*)(Args...) const> {
  FUNCTION_TRAITS_STRUCT_BODY;
  constexpr static bool is_member = true;
  constexpr static bool is_static_member = true;
  typedef ClassType class_type;
};

template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
  FUNCTION_TRAITS_STRUCT_BODY;
  constexpr static bool is_member = false;
};

} // namespace tc
