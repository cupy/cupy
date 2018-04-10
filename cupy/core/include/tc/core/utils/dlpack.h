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

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include <dlpack/dlpack.h>

//
// Various utilities for DLPack, in particular DLTensor.
//

namespace tc {
namespace dlutils {

DLContext getCPUDLContext();
DLContext getGPUDLContext(int device_id = 0);
template <typename T>
DLDataType getDLDataType();

struct DLTensorDeleter {
  inline void operator()(DLTensor* t) {
    if (t->shape) {
      delete[] t->shape;
    }
    if (t->strides) {
      delete[] t->strides;
    }
    delete t;
  }
};
typedef std::shared_ptr<DLTensor> DLTensorSPtr;
typedef std::unique_ptr<DLTensor, DLTensorDeleter> DLTensorUPtr;

void SetStridesFromSizes(DLTensor& t, const std::vector<int64_t>&);
void SetSizes(DLTensor& t, const std::vector<int64_t>& sizes);
void SetStrides(DLTensor& t, const std::vector<int64_t>& strides);
DLTensorUPtr makeDLTensorWithSizes(
    DLContext ctx,
    DLDataType dtype,
    const std::vector<int64_t>& sizes);

std::vector<const DLTensor*> extractRawPtrs(
    const std::vector<DLTensorUPtr>& uptrs);
std::vector<const DLTensor*> constPtrs(const std::vector<DLTensor*>& ptrs);

// Deep copies
DLTensorUPtr makeDLTensor(const DLTensor* ptr);

template <typename T>
std::vector<DLTensorUPtr> makeDLTensorVector(const std::vector<T*>& ptrs);

bool operator==(const DLDataType& t1, const DLDataType& t2);
std::string toString(const DLDataType& t);
std::ostream& operator<<(std::ostream& os, const DLTensor& t);
std::ostream& operator<<(std::ostream& os, const DLDataType& t);

// Shape/stride/type-only comparisons
bool compareDLTensorMetadata(const DLTensor& t1, const DLTensor& t2);
template <typename T, typename TT>
bool compareDLTensorVectorMetadata(
    const std::vector<T*>& v1,
    const std::vector<TT*>& v2);
} // namespace dlutils
} // namespace tc

#include "tc/core/utils/dlpack-inl.h"
