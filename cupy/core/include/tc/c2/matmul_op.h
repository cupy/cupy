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
#include "tc/library/matmul.h"

namespace caffe2 {

template <typename T, class Context, class Engine = caffe2::DefaultEngine>
class TcMatMulOp : public TcOp<T, Context, Engine> {
 public:
  static constexpr auto description = tc::TC_MATMUL;

  TcMatMulOp(const caffe2::OperatorDef& operator_def, caffe2::Workspace* ws)
      : TcOp<T, Context, Engine>(operator_def, ws) {
    bool trans_a = OperatorBase::GetSingleArgument<int>("trans_a", 0);
    bool trans_b = OperatorBase::GetSingleArgument<int>("trans_b", 0);
    this->tc_ = tc::makeMatmulTc(trans_a, trans_b);
    this->tcName_ = tc::TC_MATMUL_NAME;
  }

  ~TcMatMulOp() override {}

 protected:
  void setupNaiveMappingOptions() override {
    this->mappingOptions_.tile({16, 16, 32})
        .mapToThreads(4, 32)
        .mapToBlocks(32, 32, 32)
        .unroll(1);
  }
};
} // namespace caffe2
