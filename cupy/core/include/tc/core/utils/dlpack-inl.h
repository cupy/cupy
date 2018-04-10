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
namespace dlutils {

inline std::string toString(const DLDataType& t) {
  if (t.lanes != 1) {
    CHECK(false) << "NYI: toString for >1 lanes";
  }
  switch (t.code) {
    case DLDataTypeCode::kDLFloat:
      switch (t.bits) {
        case 16:
          return "Half";
        case 32:
          return "float";
        case 64:
          return "double";
      }
      break;
    case DLDataTypeCode::kDLInt:
      switch (t.bits) {
        case 8:
          return "int8_t";
        case 16:
          return "int16_t";
        case 32:
          return "int";
        case 64:
          return "int64_t";
      }
      break;
    case DLDataTypeCode::kDLUInt:
      switch (t.bits) {
        case 8:
          return "uint8_t";
      }
      break;
  }
  CHECK(false) << "NYI: toString for type: " << t.code << ", bits: " << t.bits;
  return "";
}

inline DLContext getCPUDLContext() {
  DLContext res;
  res.device_id = 0;
  res.device_type = DLDeviceType::kDLCPU;
  return res;
}

// Can't have a CUDAContext object here, howdo we get the GPU id of a C2 Tensor
// ?
inline DLContext getGPUDLContext(int device_id) {
  DLContext res;
  res.device_id = device_id;
  res.device_type = DLDeviceType::kDLGPU;
  return res;
}

template <typename T>
DLDataType getDLDataType();

template <>
inline DLDataType getDLDataType<float>() {
  DLDataType res;
  res.code = DLDataTypeCode::kDLFloat;
  res.bits = 32;
  res.lanes = 1;
  return res;
}

template <>
inline DLDataType getDLDataType<int>() {
  DLDataType res;
  res.code = DLDataTypeCode::kDLInt;
  res.bits = 32;
  res.lanes = 1;
  return res;
}

inline void SetSizes(DLTensor& t, const std::vector<int64_t>& sizes) {
  auto ndim = sizes.size();
  for (size_t i = 0; i < ndim; ++i) {
    t.shape[i] = sizes[i];
  }
}

inline void SetStrides(DLTensor& t, const std::vector<int64_t>& strides) {
  auto ndim = strides.size();
  for (size_t i = 0; i < ndim; ++i) {
    t.strides[i] = strides[i];
  }
}

inline void SetStridesFromSizes(DLTensor& t, const std::vector<int64_t>&) {
  auto ndim = t.ndim;
  t.strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) {
    t.strides[i] = t.strides[i + 1] * t.shape[i + 1];
  }
}

inline DLTensorUPtr makeDLTensorWithSizes(
    DLContext ctx,
    DLDataType dtype,
    const std::vector<int64_t>& sizes) {
  DLTensorUPtr res(new DLTensor);
  res->data = nullptr;
  res->ctx = ctx;
  auto ndim = sizes.size();
  res->ndim = ndim;
  res->dtype = dtype;
  res->shape = new int64_t[ndim];
  SetSizes(*res, sizes);
  res->strides = new int64_t[ndim];
  SetStridesFromSizes(*res, sizes);
  res->byte_offset = 0;
  return res;
}

inline std::vector<const DLTensor*> extractRawPtrs(
    const std::vector<DLTensorUPtr>& uptrs) {
  std::vector<const DLTensor*> res;
  res.reserve(uptrs.size());
  for (const auto& uptr : uptrs) {
    res.push_back(uptr.get());
  }
  return res;
}

inline std::vector<const DLTensor*> constPtrs(
    const std::vector<DLTensor*>& ptrs) {
  std::vector<const DLTensor*> res;
  res.reserve(ptrs.size());
  for (auto p : ptrs) {
    res.push_back(p);
  }
  return res;
}

inline DLTensorUPtr makeDLTensor(const DLTensor* ptr) {
  auto res = DLTensorUPtr(new DLTensor);
  // DLTensor is not owning, so just copy the pointer
  res->data = ptr->data;
  res->ctx = ptr->ctx;
  res->ndim = ptr->ndim;
  res->dtype = ptr->dtype;

  res->shape = new int64_t[ptr->ndim];
  for (int i = 0; i < ptr->ndim; ++i) {
    res->shape[i] = ptr->shape[i];
  }
  if (ptr->strides) {
    res->strides = new int64_t[ptr->ndim];
    for (int i = 0; i < ptr->ndim; ++i) {
      res->strides[i] = ptr->strides[i];
    }
  } else {
    res->strides = NULL;
  }
  res->byte_offset = ptr->byte_offset;
  return res;
}

template <typename T>
std::vector<DLTensorUPtr> makeDLTensorVector(const std::vector<T*>& ptrs) {
  std::vector<DLTensorUPtr> res;
  for (auto p : ptrs) {
    res.push_back(makeDLTensor(p));
  }
  return res;
}

inline std::string toString(const DLTensor& t) {
  std::stringstream ss;
  ss << "DLTensor(@" << t.data << ", dim=" << t.ndim << ")";
  ss << " shape [";
  for (int i = 0; i < t.ndim; ++i) {
    ss << t.shape[i];
    if (i < t.ndim - 1) {
      ss << ", ";
    }
  }
  ss << "], strides [";
  for (int i = 0; i < t.ndim; ++i) {
    ss << t.strides[i];
    if (i < t.ndim - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

inline bool operator==(const DLDataType& t1, const DLDataType& t2) {
  return t1.code == t2.code && t1.bits == t2.bits && t1.lanes == t2.lanes;
}

inline std::ostream& operator<<(std::ostream& os, const DLDataType& t) {
  return os << "typecode: " << t.code << " bits: " << t.bits
            << " lanes: " << t.lanes;
}

inline bool compareDLTensorMetadata(const DLTensor& t1, const DLTensor& t2) {
  if (t1.ndim != t2.ndim) {
    return false;
  }
  if (!(t1.dtype == t2.dtype)) {
    return false;
  }
  if ((t1.strides == NULL) ^ (t2.strides == NULL)) {
    return false;
  }
  for (int i = 0; i < t1.ndim; ++i) {
    if (t1.shape[i] != t2.shape[i]) {
      return false;
    }
    if (t1.strides && t1.strides[i] != t2.strides[i]) {
      return false;
    }
  }
  return true;
}

// templating to have any combination of const/non-const
template <typename T, typename TT>
bool compareDLTensorVectorMetadata(
    const std::vector<T*>& v1,
    const std::vector<TT*>& v2) {
  if (v1.size() != v2.size()) {
    return false;
  }
  for (size_t i = 0; i < v1.size(); ++i) {
    if (!compareDLTensorMetadata(*v1[i], *v2[i])) {
      return false;
    }
  }
  return true;
}
} // namespace dlutils
} // namespace tc
