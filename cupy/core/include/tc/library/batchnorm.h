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

static constexpr auto TC_BATCHNORM_NAME = "spatialBatchNorm";

static constexpr auto TC_BATCHNORM = R"TC(
   def spatialBatchNorm(
     float momentum, float eps,
     float(N,C,H,W) I, float(C) rMeanIn, float(C) rVarIn)
   -> (O, rMeanOut, rVarOut, mean, centered, variance, expectedVariance, normalizedOut)
   {
      mean(c) +=! I(nn, c, hh, ww)
      mean(c)  = mean(c) / (N * H * W)
      rMeanOut(c) = (1 - momentum) * rMeanIn(c) + momentum * mean(c)
      centered(n, c, h, w) = I(n, c, h, w) - rMeanOut(c)
      variance(n, c, h, w) = centered(n, c, h, w) * centered(n, c, h, w)
      expectedVariance(c) +=! (variance(n, c, h, w) + eps) / (N * H * W)
      rVarOut(c) = rsqrt(
        (1 - momentum) * rVarIn(c) + momentum * expectedVariance(c))
      O(n, c, h, w) = centered(n, c, h, w) * rVarOut(c)
      normalizedOut(n, c, h, w) = O(n, c, h, w)
   })TC";

} // namespace tc
