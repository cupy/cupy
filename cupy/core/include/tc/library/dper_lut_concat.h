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

constexpr static auto TC_DPER_LUT_CONCAT_NAME = "dper_lut_concat";

constexpr static auto TC_DPER_LUT_CONCAT = R"TC(
  def dper_lut_concat(float(B, M) I1, int32(B, L1) I2, int32(B, L2) I3, float(N, M) W1, float(N) B1, float(E1, D) LUT1, float(E2, D) LUT2) -> (O1, O2, O3) {
    O1(b, n) +=! I1(b, m) * W1(n, m)
    O1(b, n) = O1(b, n) + B1(n)
    O2(i, j) +=! LUT1(I2(i, k), j)
    O3(i, j) +=! LUT2(I3(i, k), j)
  }
)TC";

} // namespace tc
