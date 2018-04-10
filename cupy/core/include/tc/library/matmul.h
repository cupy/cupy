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

static constexpr auto TC_MATMUL_NAME = "matmul";

namespace {
static constexpr auto TC_MATMUL = R"TC(
  def matmul(float(${szA0}, ${szA1}) A, float(${szB0}, ${szB1}) B) -> (O) {
    O(i, j) +=! A(${itA0}, ${itA1}) * B(${itB0}, ${itB1})
  }
)TC";
} // namespace

std::string makeMatmulTc(
    bool transposeFirst = false,
    bool transposeSecond = false) {
  std::string tc(TC_MATMUL);
  tc = replaceString(tc, "${szA0}", (transposeFirst ? "K" : "N"));
  tc = replaceString(tc, "${szA1}", (transposeFirst ? "N" : "K"));
  tc = replaceString(tc, "${szB0}", (transposeSecond ? "M" : "K"));
  tc = replaceString(tc, "${szB1}", (transposeSecond ? "K" : "M"));
  tc = replaceString(tc, "${itA0}", (transposeFirst ? "k" : "i"));
  tc = replaceString(tc, "${itA1}", (transposeFirst ? "i" : "k"));
  tc = replaceString(tc, "${itB0}", (transposeSecond ? "j" : "k"));
  tc = replaceString(tc, "${itB1}", (transposeSecond ? "k" : "j"));
  return tc;
}
} // namespace tc
