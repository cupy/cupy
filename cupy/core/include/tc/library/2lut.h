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

namespace tc {

constexpr static auto TC_2LUT_NAME = "func_2lut";

constexpr static auto TC_2LUT = R"TC(
  def func_2lut(float(E1, D) LUT1, int32(B, L1) I1, float(E2, D) LUT2, int32(B, L2) I2) -> (O1, O2) {
    O1(i, j) +=! LUT1(I1(i, k), j)
    O2(i, j) +=! LUT2(I2(i, k), j)
  }
)TC";

} // namespace tc
