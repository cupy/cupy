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

#include <iostream>
#include <string>

#include "tc/core/mapping_options.h"

namespace tc {

class MappingOptionsAsCpp {
 public:
  explicit MappingOptionsAsCpp(
      const MappingOptions& options_,
      size_t indent_ = 0)
      : options(options_), indent(indent_) {}
  const MappingOptions& options;
  size_t indent;
};

class MappingOptionsCppPrinter {
 public:
  MappingOptionsCppPrinter(std::ostream& out, size_t ws = 0)
      : out_(out), ws_(ws) {}

  friend MappingOptionsCppPrinter& operator<<(
      MappingOptionsCppPrinter& prn,
      const std::string& str);

  friend MappingOptionsCppPrinter& operator<<(
      MappingOptionsCppPrinter& prn,
      const MappingOptions& options);

 private:
  inline MappingOptionsCppPrinter& tab();
  inline MappingOptionsCppPrinter& endl();
  inline MappingOptionsCppPrinter& endStmt();
  inline MappingOptionsCppPrinter& printString(const char* str);
  inline MappingOptionsCppPrinter& printString(const std::string str) {
    return printString(str.c_str());
  }
  inline MappingOptionsCppPrinter& printBooleanOption(
      const std::string& name,
      bool value);

  template <typename T>
  inline MappingOptionsCppPrinter& printListOption(
      const std::string& name,
      const std::vector<T>& values);

  template <typename T>
  inline MappingOptionsCppPrinter& printValueOption(
      const std::string& name,
      const T& value);

  MappingOptionsCppPrinter& printSchedulerOptions(
      const SchedulerOptionsView& schedulerOptions,
      const std::string& prefix);

  std::ostream& out_;
  size_t ws_;
  bool lineContinuation_ = false;
};

inline std::ostream& operator<<(
    std::ostream& out,
    const MappingOptionsAsCpp& mo) {
  auto prn = MappingOptionsCppPrinter(out, mo.indent);
  prn << mo.options;
  return out;
}

MappingOptionsCppPrinter& MappingOptionsCppPrinter::tab() {
  for (size_t i = 0; i < ws_; ++i) {
    out_ << " ";
  }
  return *this;
}

MappingOptionsCppPrinter& MappingOptionsCppPrinter::endl() {
  this->out_ << std::endl;
  if (!lineContinuation_) {
    lineContinuation_ = true;
    ws_ += 4;
  }
  return *this;
}

MappingOptionsCppPrinter& MappingOptionsCppPrinter::endStmt() {
  this->out_ << ";" << std::endl;
  lineContinuation_ = false;
  ws_ -= 4;
  return *this;
}

MappingOptionsCppPrinter& MappingOptionsCppPrinter::printString(
    const char* str) {
  this->out_ << str;
  return *this;
}

MappingOptionsCppPrinter& MappingOptionsCppPrinter::printBooleanOption(
    const std::string& name,
    bool value) {
  endl();
  tab();
  out_ << "." << name << "(" << (value ? "true" : "false") << ")";
  return *this;
}

template <typename T>
MappingOptionsCppPrinter& MappingOptionsCppPrinter::printValueOption(
    const std::string& name,
    const T& value) {
  endl();
  tab();
  out_ << "." << name << "(" << value << ")";
  return *this;
}

template <typename T>
MappingOptionsCppPrinter& MappingOptionsCppPrinter::printListOption(
    const std::string& name,
    const std::vector<T>& values) {
  endl();
  tab();
  out_ << "." << name << "(";
  if (values.size() > 0) {
    out_ << values[0];
  }
  for (size_t i = 1, e = values.size(); i < e; ++i) {
    out_ << ", " << values[i];
  }
  out_ << ")";
  return *this;
}
} // namespace tc
