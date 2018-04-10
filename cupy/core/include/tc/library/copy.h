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

#include "tc/library/common.h"

namespace tc {

static constexpr auto COPY_DOC = R"DOC(
  def copy(float(...) I) -> (O) {
    O(...) = I(...)
  }
)DOC";

constexpr static auto COPY_TC_NAME = "copy";

constexpr static auto COPY_GRAD_TC_NAME = "copyGrad";

namespace {
constexpr static auto COPY_TC = R"TC(
  def copy(float(${dimParams}) I) -> (O) {
    O(${dimIndices}) = I(${dimIndices})
  }
)TC";

constexpr static auto COPY_GRAD_TC = R"TC(
  def copyGrad(float(${dimParams}) O_grad) -> (I_grad) {
    I_grad(${dimIndices}) = O_grad(${dimIndices})
  }
)TC";
} // namespace

std::string
setInputDims(std::string tcStr, int numDims, std::string paramPrefix) {
  std::string dimParams, dimIndices;
  for (int i = 0; i < numDims; i++) {
    dimParams += paramPrefix + std::to_string(i);
    dimIndices += "i" + std::to_string(i);
    if (i < numDims - 1) {
      dimParams += ",";
      dimIndices += ",";
    }
  }
  tcStr = replaceString(tcStr, "${dimParams}", dimParams);
  tcStr = replaceString(tcStr, "${dimIndices}", dimIndices);
  return tcStr;
}

std::string makeCopyTc(int numDims) {
  return setInputDims(COPY_TC, numDims, "P");
}

std::string makeCopyGradTc(int numDims) {
  return setInputDims(COPY_GRAD_TC, numDims, "PG");
}

} // namespace tc
