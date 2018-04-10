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

#include "tc/lang/tree.h"

namespace lang {

struct ErrorReport : public std::exception {
  ErrorReport(const ErrorReport& e)
      : ss(e.ss.str()), context(e.context), the_message(e.the_message) {}

  ErrorReport(TreeRef context) : context(context->range()) {}
  ErrorReport(SourceRange range) : context(std::move(range)) {}
  virtual const char* what() const noexcept override {
    std::stringstream msg;
    msg << "\n" << ss.str() << ":\n";
    context.highlight(msg);
    the_message = msg.str();
    return the_message.c_str();
  }

 private:
  template <typename T>
  friend const ErrorReport& operator<<(const ErrorReport& e, const T& t);

  mutable std::stringstream ss;
  SourceRange context;
  mutable std::string the_message;
};

inline void warn(const ErrorReport& err) {
  std::cerr << err.what();
}

template <typename T>
const ErrorReport& operator<<(const ErrorReport& e, const T& t) {
  e.ss << t;
  return e;
}

#define TC_ASSERT(ctx, cond)                                               \
  if (!(cond)) {                                                           \
    throw ::lang::ErrorReport(ctx)                                         \
        << __FILE__ << ":" << __LINE__ << ": assertion failed: " << #cond; \
  }
} // namespace lang
