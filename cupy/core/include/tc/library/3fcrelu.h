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

constexpr static auto TC_3FCRELU_NAME = "func_3fcrelu";

constexpr static auto TC_3FCRELU = R"TC(
  def func_3fcrelu(float(B,M) I, float(N,M) W1, float(N) B1, float(O,N) W2, float(O) B2, float(P,O) W3, float(P) B3) -> (O1, O2, O3) {
    O1(b, n) +=! I(b, m) * W1(n, m)
    O1(b, n) = O1(b, n) + B1(n)
    O1(b, n) = fmax(O1(b, n), 0)
    O2(b, o) +=! O1(b, n) * W2(o, n)
    O2(b, o) = O2(b, o) + B2(o)
    O2(b, o) = fmax(O2(b, o), 0)
    O3(b, p) +=! O2(b, o) * W3(p, o)
    O3(b, p) = O3(b, p) + B3(p)
    O3(b, p) = fmax(O3(b, p), 0)
  }
)TC";

} // namespace tc
