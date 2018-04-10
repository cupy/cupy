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

#include "tc/core/utils/vararg.h"

namespace tc {

//
// TilingView & Tiling
//

Tiling::Tiling(const std::vector<uint64_t>& sizes) : TilingView(ownedProto_) {
  proto.clear_sizes();
  std::copy(
      sizes.begin(),
      sizes.end(),
      google::protobuf::RepeatedFieldBackInserter(proto.mutable_sizes()));
}

Tiling::Tiling(std::initializer_list<uint64_t> il)
    : Tiling(std::vector<uint64_t>(il)) {}

std::vector<uint64_t> TilingView::extractVector() const {
  std::vector<uint64_t> result(proto.sizes().begin(), proto.sizes().end());
  return result;
}

size_t TilingView::size() const {
  return proto.sizes_size();
}

ValueAccessor<uint64_t> TilingView::operator[](size_t i) {
  CHECK_LT(i, proto.sizes_size()) << "index overflow";
  return ValueAccessor<uint64_t>(
      [this, i](uint64_t u) { this->proto.set_sizes(i, u); },
      [this, i]() { return this->proto.sizes(i); });
}

uint64_t TilingView::operator[](size_t i) const {
  CHECK_LT(i, proto.sizes_size()) << "index overflow";
  return proto.sizes(i);
}

TilingView& TilingView::operator=(const TilingView& view) {
  proto = view.proto;
  return *this;
}

bool TilingView::operator==(const TilingView& view) const {
  return proto.SerializeAsString() == view.proto.SerializeAsString();
}

bool TilingView::operator!=(const TilingView& view) const {
  return !(*this == view);
}

//
// CudaDimView & CudaDim
//
CudaDim::CudaDim(std::vector<uint64_t> il) : CudaDimView(ownedProto_) {
  CHECK_GT(il.size(), 0) << "list of values in CudaDimView must be non-empty";
  CHECK_LE(il.size(), 3) << "at most 3 values allowed in CudaDimView";

  switch (il.size()) {
    case 3:
      proto.set_z(*(il.begin() + 2));
    case 2:
      proto.set_y(*(il.begin() + 1));
    case 1:
      proto.set_x(*il.begin());
      break;
    default:
      CHECK(false) << "unreachable";
  }
}

CudaDim::CudaDim(std::initializer_list<uint64_t> il)
    : CudaDim(std::vector<uint64_t>(il)) {}

CudaDim::CudaDim(uint64_t x, uint64_t y, uint64_t z)
    : CudaDimView(ownedProto_) {
  proto.set_x(x);
  if (y != defaultDim || z != defaultDim) {
    proto.set_y(y);
  }
  if (z != defaultDim) {
    proto.set_z(z);
  }
}

size_t CudaDimView::size() const {
  CHECK(!(!proto.has_y() && proto.has_z())) << "CudaDimView has z but not y";

  if (proto.has_z() && proto.has_y()) {
    return 3;
  } else if (proto.has_y()) {
    return 2;
  }
  return 1;
}

std::vector<uint64_t> CudaDimView::extractVector() const {
  CHECK(!(!proto.has_y() && proto.has_z())) << "CudaDimView has z but not y";

  std::vector<uint64_t> result;
  result.push_back(proto.x());
  if (proto.has_y()) {
    result.push_back(proto.y());
  }
  if (proto.has_z()) {
    result.push_back(proto.z());
  }
  return result;
}

std::array<uint64_t, 3> CudaDimView::extractDefaultedArray() const {
  std::array<uint64_t, 3> arr{CudaDimView::defaultDim,
                              CudaDimView::defaultDim,
                              CudaDimView::defaultDim};
  auto v = extractVector();
  CHECK_LE(v.size(), 3);
  std::copy(v.begin(), v.end(), arr.begin());
  return arr;
}

ValueAccessor<uint64_t> CudaDimView::operator[](size_t i) {
  CHECK_LT(i, 3) << "index overflow";
  if (i == 0) {
    return ValueAccessor<uint64_t>(
        [this](uint64_t u) { this->proto.set_x(u); },
        [this]() { return this->proto.x(); });
  } else if (i == 1) {
    return ValueAccessor<uint64_t>(
        [this](uint64_t u) { this->proto.set_y(u); },
        [this]() {
          return this->proto.has_y() ? this->proto.y()
                                     : CudaDimView::defaultDim;
        });
  } else {
    return ValueAccessor<uint64_t>(
        [this](uint64_t u) { this->proto.set_z(u); },
        [this]() {
          return this->proto.has_z() ? this->proto.z()
                                     : CudaDimView::defaultDim;
        });
  }
}

uint64_t CudaDimView::operator[](size_t i) const {
  CHECK_LT(i, 3) << "index overflow";
  if (i == 0) {
    return proto.x();
  } else if (i == 1) {
    return proto.has_y() ? proto.y() : CudaDimView::defaultDim;
  } else {
    return proto.has_z() ? proto.z() : CudaDimView::defaultDim;
  }
}

CudaDimView& CudaDimView::operator=(const CudaDimView& view) {
  proto = view.proto;
  return *this;
}

bool CudaDimView::operator==(const CudaDimView& view) const {
  return proto.SerializeAsString() == view.proto.SerializeAsString();
}

bool CudaDimView::operator!=(const CudaDimView& view) const {
  return !(*this == view);
}

//
// SchedulerOptionsView & SchedulerOptions
//
SchedulerOptionsView& SchedulerOptionsView::operator=(
    const SchedulerOptionsView& view) {
  proto = view.proto;
  return *this;
}

bool SchedulerOptionsView::operator==(const SchedulerOptionsView& view) const {
  return proto.SerializeAsString() == view.proto.SerializeAsString();
}

bool SchedulerOptionsView::operator!=(const SchedulerOptionsView& view) const {
  return !(*this == view);
}

//
// MappingOptions
//
MappingOptions::MappingOptions()
    : block(*proto.mutable_block()),
      grid(*proto.mutable_grid()),
      tiling(*proto.mutable_tiling()),
      outerScheduleOptions(*proto.mutable_outer_schedule_options()),
      intraTileScheduleOptions(*proto.mutable_intra_tile_schedule_options()) {}

MappingOptions::MappingOptions(const MappingOptions& options)
    : proto(options.proto),
      block(*proto.mutable_block()),
      grid(*proto.mutable_grid()),
      tiling(*proto.mutable_tiling()),
      outerScheduleOptions(*proto.mutable_outer_schedule_options()),
      intraTileScheduleOptions(*proto.mutable_intra_tile_schedule_options()) {}

MappingOptions::MappingOptions(const MappingOptionsProto& buf)
    : proto(buf),
      block(*proto.mutable_block()),
      grid(*proto.mutable_grid()),
      tiling(*proto.mutable_tiling()),
      outerScheduleOptions(*proto.mutable_outer_schedule_options()),
      intraTileScheduleOptions(*proto.mutable_intra_tile_schedule_options()) {}

MappingOptions::MappingOptions(const std::string& str) : MappingOptions() {
  bool parsed = proto.ParseFromString(str);
  CHECK(parsed) << "could not parse protobuf string";
}

bool MappingOptions::operator==(const MappingOptions& options) const {
  return proto.SerializeAsString() == options.proto.SerializeAsString();
}

bool MappingOptions::operator!=(const MappingOptions& options) const {
  return !(*this == options);
}

std::string MappingOptions::toProtobufSerializedString() const {
  return proto.SerializeAsString();
}

//
// MappingOptions chainable builders.
//

MappingOptions& MappingOptions::tile(const std::vector<uint64_t>& sizes) {
  tiling = Tiling(sizes);
  return *this;
}

MappingOptions& MappingOptions::tile(std::initializer_list<uint64_t> sizes) {
  tiling = Tiling(sizes);
  return *this;
}

MappingOptions& MappingOptions::tile(const char* str) {
  return tile(std::string(str));
}

template <typename... Args>
MappingOptions& MappingOptions::tile(Args... args) {
  static_assert(
      TemplArgsAll<std::is_integral, Args...>::value,
      "arguments of tile() must be integers");
  return tile(vectorFromCastedArgs<uint64_t, Args...>(args...));
}

MappingOptions& MappingOptions::mapToThreads(
    std::initializer_list<uint64_t> threads) {
  block = CudaDim(threads);
  return *this;
}

MappingOptions&
MappingOptions::mapToThreads(uint64_t x, uint64_t y, uint64_t z) {
  block = CudaDim(x, y, z);
  return *this;
}

MappingOptions& MappingOptions::mapToThreads(
    const std::vector<uint64_t>& threads) {
  CHECK_GT(threads.size(), 0) << "expected at least one thread size";
  CHECK_LE(threads.size(), 3) << "expected at most three thread sizes";

  uint64_t x = threads[0];
  uint64_t y = threads.size() > 1 ? threads[1] : CudaDimView::defaultDim;
  uint64_t z = threads.size() > 2 ? threads[2] : CudaDimView::defaultDim;
  block = CudaDim(x, y, z);
  return *this;
}

MappingOptions& MappingOptions::mapToBlocks(
    std::initializer_list<uint64_t> blocks) {
  grid = CudaDim(blocks);
  return *this;
}

MappingOptions&
MappingOptions::mapToBlocks(uint64_t x, uint64_t y, uint64_t z) {
  grid = CudaDim(x, y, z);
  return *this;
}

MappingOptions& MappingOptions::mapToBlocks(
    const std::vector<uint64_t>& blocks) {
  CHECK_GT(blocks.size(), 0) << "expected at least one thread size";
  CHECK_LE(blocks.size(), 3) << "expected at most three thread sizes";

  uint64_t x = blocks[0];
  uint64_t y = blocks.size() > 1 ? blocks[1] : CudaDimView::defaultDim;
  uint64_t z = blocks.size() > 2 ? blocks[2] : CudaDimView::defaultDim;
  grid = CudaDim(x, y, z);
  return *this;
}

MappingOptions& MappingOptions::unroll(uint64_t size) {
  proto.set_unroll(size);
  return *this;
}

MappingOptions& MappingOptions::useSharedMemory(bool b) {
  proto.set_use_shared_memory(b);
  return *this;
}

MappingOptions& MappingOptions::usePrivateMemory(bool b) {
  proto.set_use_private_memory(b);
  return *this;
}

MappingOptions& MappingOptions::maxSharedMemory(uint64_t size) {
  proto.set_max_shared_memory(size);
  return *this;
}

MappingOptions& MappingOptions::fixParametersBeforeScheduling(bool b) {
  proto.set_fix_parameters_before_scheduling(b);
  return *this;
}

MappingOptions& MappingOptions::unrollCopyShared(bool b) {
  proto.set_unroll_copy_shared(b);
  return *this;
}

MappingOptions& MappingOptions::tileImperfectlyNested(bool b) {
  proto.set_tile_imperfectly_nested(b);
  return *this;
}

MappingOptions& MappingOptions::matchLibraryCalls(bool b) {
  proto.set_match_library_calls(b);
  return *this;
}

MappingOptions& MappingOptions::scheduleFusionStrategy(FusionStrategy fs) {
  outerScheduleFusionStrategy(fs);
  intraTileScheduleFusionStrategy(fs);
  return *this;
}

MappingOptions& MappingOptions::scheduleFusionStrategy(const std::string& str) {
  FusionStrategy fs;
  bool couldParse = FusionStrategy_Parse(str, &fs);
  CHECK(couldParse) << "unknown FusionStrategy " << str;
  return scheduleFusionStrategy(fs);
}

MappingOptions& MappingOptions::outerScheduleFusionStrategy(FusionStrategy fs) {
  outerScheduleOptions.proto.set_fusion_strategy(fs);
  return *this;
}

MappingOptions& MappingOptions::outerScheduleFusionStrategy(
    const std::string& str) {
  FusionStrategy fs;
  bool couldParse = FusionStrategy_Parse(str, &fs);
  CHECK(couldParse) << "unknown FusionStrategy " << str;
  return outerScheduleFusionStrategy(fs);
}

MappingOptions& MappingOptions::outerScheduleAllowSkewing(bool b) {
  outerScheduleOptions.proto.set_allow_skewing(b);
  return *this;
}

MappingOptions& MappingOptions::outerSchedulePositiveOrthant(bool b) {
  outerScheduleOptions.proto.set_positive_orthant(b);
  return *this;
}

MappingOptions& MappingOptions::intraTileScheduleFusionStrategy(
    FusionStrategy fs) {
  intraTileScheduleOptions.proto.set_fusion_strategy(fs);
  return *this;
}

MappingOptions& MappingOptions::intraTileScheduleFusionStrategy(
    const std::string& str) {
  FusionStrategy fs;
  bool couldParse = FusionStrategy_Parse(str, &fs);
  CHECK(couldParse) << "unknown FusionStrategy " << str;
  return intraTileScheduleFusionStrategy(fs);
}

MappingOptions& MappingOptions::intraTileScheduleAllowSkewing(bool b) {
  intraTileScheduleOptions.proto.set_allow_skewing(b);
  return *this;
}

MappingOptions& MappingOptions::intraTileSchedulePositiveOrthant(bool b) {
  intraTileScheduleOptions.proto.set_positive_orthant(b);
  return *this;
}

} // namespace tc
