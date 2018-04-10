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

static constexpr auto TC_BATCH_MATMUL_NAME = "batch_matmul";

static constexpr auto TC_BATCH_MATMUL = R"TC(
   def batch_matmul(float(B, N, M) X, float(B, M, K) Y) -> (Z) {
     Z(b, n, k) +=! X(b, n, mm) * Y(b, mm, k)
   }
)TC";

} // namespace tc
