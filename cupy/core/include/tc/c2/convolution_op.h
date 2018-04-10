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

#include <string>
#include <vector>

#include "tc/c2/tc_op.h"
#include "tc/library/convolution.h"

namespace caffe2 {

template <typename T, class Context, class Engine = caffe2::DefaultEngine>
class TcConvolutionOp : public TcOp<T, Context, Engine> {
 public:
  static constexpr auto description = tc::CONVOLUTION2D_TC;

  TcConvolutionOp(
      const caffe2::OperatorDef& operator_def,
      caffe2::Workspace* ws)
      : TcOp<T, Context, Engine>(operator_def, ws) {
    int strideH = 0;
    int strideW = 0;
    if (OperatorBase::HasArgument("stride")) {
      strideH = OperatorBase::GetSingleArgument<int>("stride", 1);
      strideW = OperatorBase::GetSingleArgument<int>("stride", 1);
    } else {
      strideH = OperatorBase::GetSingleArgument<int>("stride_h", 1);
      strideW = OperatorBase::GetSingleArgument<int>("stride_w", 1);
    }

    int padT = 0;
    int padL = 0;
    int padB = 0;
    int padR = 0;
    if (OperatorBase::HasArgument("pad")) {
      padT = OperatorBase::GetSingleArgument<int>("pad", 0);
      padL = OperatorBase::GetSingleArgument<int>("pad", 0);
      padB = OperatorBase::GetSingleArgument<int>("pad", 0);
      padR = OperatorBase::GetSingleArgument<int>("pad", 0);
    } else {
      padT = OperatorBase::GetSingleArgument<int>("pad_t", 0);
      padL = OperatorBase::GetSingleArgument<int>("pad_l", 0);
      padB = OperatorBase::GetSingleArgument<int>("pad_b", 0);
      padR = OperatorBase::GetSingleArgument<int>("pad_r", 0);
    }

    CHECK(padT == 0 && padL == 0 && padB == 0 && padR == 0)
        << "NYI: padding larger than 0";

    this->tc_ = tc::makeConvolution2DTc(strideH, strideW);
    this->tcName_ = tc::CONVOLUTION2D_TC_NAME;
    this->gradTc_ = tc::makeConvolution2DGradTc(strideH, strideW);
    this->gradTcName_ = tc::CONVOLUTION2D_GRAD_TC_NAME;
  }

  ~TcConvolutionOp() override {}

 protected:
  void setupNaiveMappingOptions() override {
    this->mappingOptions_ = tc::MappingOptions::makeConvolutionMappingOptions();
    this->gradMappingOptions_ =
        tc::MappingOptions::makeConvolutionMappingOptions();
  }
};
} // namespace caffe2
