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
#include "tc/library/2lut.h"

namespace caffe2 {

template <
    typename T,
    typename I,
    class Context,
    class Engine = caffe2::DefaultEngine>
class Tc2LUTOp : public TcOp<T, Context, Engine> {
 public:
  static constexpr auto description = tc::TC_2LUT;

  Tc2LUTOp(const caffe2::OperatorDef& operator_def, caffe2::Workspace* ws)
      : TcOp<T, Context, Engine>(operator_def, ws) {
    this->tc_ = tc::TC_2LUT;
    this->tcName_ = tc::TC_2LUT_NAME;
  }

  ~Tc2LUTOp() override {}

 protected:
  void setupNaiveMappingOptions() override {
    this->mappingOptions_.mapToBlocks(256)
        .mapToThreads(64)
        .tile({1})
        .unroll(1)
        .useSharedMemory(false)
        .usePrivateMemory(false);
  }
};
} // namespace caffe2
