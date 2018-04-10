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

#include <stdexcept>
#include <string>

namespace tc {
namespace polyhedral {

struct EmptyFilterException : public std::runtime_error {
  explicit EmptyFilterException(const std::string& s) : std::runtime_error(s) {}
};

struct EmptyMappingFilterException : public std::runtime_error {
  explicit EmptyMappingFilterException(const std::string& s)
      : std::runtime_error(s) {}
};

struct NoBandsException : public std::runtime_error {
  explicit NoBandsException(const std::string& s) : std::runtime_error(s) {}
};

namespace tightening {
struct TighteningException : public std::logic_error {
  explicit TighteningException(const std::string& s)
      : std::logic_error(std::string("[TighteningException]") + s) {}
};
} // namespace tightening

namespace promotion {
struct OutOfRangeException : public std::out_of_range {
  explicit OutOfRangeException(const std::string& s) : std::out_of_range(s) {}
};

struct PromotionLogicError : public std::logic_error {
  explicit PromotionLogicError(const std::string& s) : std::logic_error(s) {}
};

struct PromotionBelowThreadsException : public PromotionLogicError {
  explicit PromotionBelowThreadsException(const std::string& s)
      : PromotionLogicError(s) {}
};

struct PromotionNYI : public std::logic_error {
  explicit PromotionNYI(const std::string& s) : std::logic_error(s) {}
};

struct GroupingError : public std::logic_error {
  explicit GroupingError(const std::string& s) : std::logic_error(s) {}
};
} // namespace promotion

} // namespace polyhedral
} // namespace tc
