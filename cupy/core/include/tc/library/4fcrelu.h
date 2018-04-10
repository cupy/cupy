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
constexpr static auto TC_4FCRELU_NAME = "func_4fcrelu";

constexpr static auto TC_4FCRELU = R"TC(
  def func_4fcrelu(float(B,M) I, float(N,M) W1, float(N) B1, float(O,N) W2, float(O) B2, float(P,O) W3, float(P) B3, float(Q,P) W4, float(Q) B4) -> (O1, O2, O3, O4) {
    O1(b, n) +=! I(b, m) * W1(n, m)
    O1(b, n) = O1(b, n) + B1(n)
    O1(b, n) = fmax(O1(b, n), 0)
    O2(b, o) +=! O1(b, n) * W2(o, n)
    O2(b, o) = O2(b, o) + B2(o)
    O2(b, o) = fmax(O2(b, o), 0)
    O3(b, p) +=! O2(b, o) * W3(p, o)
    O3(b, p) = O3(b, p) + B3(p)
    O3(b, p) = fmax(O3(b, p), 0)
    O4(b, q) +=! O3(b, p) * W4(q, p)
    O4(b, q) = O4(b, q) + B4(q)
    O4(b, q) = fmax(O4(b, q), 0)
  }
)TC";
} // namespace tc
