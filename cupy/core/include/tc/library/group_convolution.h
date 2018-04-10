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

constexpr static auto GROUP_CONVOLUTION2D_TC_NAME = "group_convolution";
constexpr static auto GROUP_CONVOLUTION2D_GRAD_TC_NAME =
    "group_convolution2dGrad";

namespace {
constexpr static auto GROUP_CONVOLUTION2D_TC = R"TC(
  def group_convolution(float(N,G,C,H,W) I, float(G,F,C,KH,KW) W1, float(G,F) B)
  -> (O)
  {
    O(n, g, f, h, w) +=!
      I(n, g, c, <sh> * h + kh, <sw> * w + kw) * W1(g, f, c, kh, kw)
    O(n, g, f, h, w) = O(n, g, f, h, w) + B(g, f)
  }
)TC";

constexpr static auto GROUP_CONVOLUTION2D_GRAD_TC = R"TC(
  def group_convolution2dGrad(float(N,G,C,H,W) I, float(G,F,C,KH,KW) W1,
                              float(N,F,H,W) O_grad)
  -> (I_grad, W1_grad, B_grad)
  {
    I_grad(n, g, c, h, w) +=!
      O_grad(n, g, f, <sh> * h - kh, <sw> * w - kw) * W1(g, f, c, kh, kw)
    W1_grad(g, f, c, kh, kw) +=!
      O_grad(n, g, f, <sh> * h - kh, <sw> * w - kw) * I(n, g, c, h, w)
    B_grad(g, f) +=! O_grad(n, g, f, h, w)
  }
)TC";
} // namespace

std::string makeGroupConvolution2DTc(int strideH, int strideW) {
  CHECK(strideH > 0 && strideW > 0) << "Stride must be greater than 0";
  std::string tcStr;
  tcStr = GROUP_CONVOLUTION2D_TC;
  tcStr = replaceString(tcStr, "<sh>", std::to_string(strideH));
  tcStr = replaceString(tcStr, "<sw>", std::to_string(strideW));
  return tcStr;
}

std::string makeGroupConvolution2DGradTc(int strideH, int strideW) {
  CHECK(strideH > 0 && strideW > 0) << "Stride must be greater than 0";
  std::string tcStr;
  tcStr = GROUP_CONVOLUTION2D_GRAD_TC;
  tcStr = replaceString(tcStr, "<sh>", std::to_string(strideH));
  tcStr = replaceString(tcStr, "<sw>", std::to_string(strideW));
  return tcStr;
}
} // namespace tc
