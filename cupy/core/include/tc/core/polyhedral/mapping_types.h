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

#include "tc/external/isl.h"

namespace tc {

struct Block;
struct Grid;

namespace polyhedral {
namespace mapping {

struct BlockId;
struct ThreadId;

struct MappingId : public isl::id {
 protected:
  MappingId(isl::id i, unsigned char l) : isl::id(i), dim(l) {}

 public:
  MappingId(const MappingId& id) : isl::id(id), dim(id.dim) {}

  inline bool isBlockId();
  inline BlockId* asBlockId();

  inline bool isThreadId();
  inline ThreadId* asThreadId();

  // For indexing into positional arrays
  // TODO: this should go away but this probably requires tinkering with
  // mapping_options.h::Grid/Block.
  // Also, generally can't have fully static types and dynamic behavior
  // like is used in mapped_scop.cc, so pick your poison:
  //   API bloat/templates or dynamic checks
  const unsigned char dim;

  // Placeholder value to use in absence of mapping size.
  static constexpr size_t unmapped = 1;

  struct Hash {
    size_t operator()(const MappingId& id) const {
      return isl::IslIdIslHash().operator()(id);
    }
  };
};

// Note: do not add members to ThreadId or slicing will ensue.
// We use containers of MappingId. This makes sense because of isl::id
// semantics and the fact that a MappingId **isa** isl::id.
struct BlockId : public MappingId {
 public:
  inline BlockId(isl::id id, unsigned char l) : MappingId(id, l) {}
  static inline BlockId makeId(size_t dim);
  template <unsigned char Dim>
  static BlockId makeId(isl::id id);

  static constexpr unsigned char kMaxDim = 3;

  static inline BlockId x();
  static inline BlockId y();
  static inline BlockId z();
  /*
   * Returns the size of the mapping or Mapping::unmapped if not mapped.
   * This uses traditional internal assumptions that x<=>0, x<=>0, y<=>1, z<=>2.
   */
  size_t mappingSize(const tc::Grid& grid) const;

  struct Hash {
    size_t operator()(const BlockId& id) const {
      return isl::IslIdIslHash().operator()(id);
    }
  };
};

// Note: do not add members to ThreadId or slicing will ensue.
// We use containers of MappingId. This makes sense because of isl::id
// semantics and the fact that a MappingId **isa** isl::id.
struct ThreadId : public MappingId {
 public:
  inline ThreadId(isl::id id, unsigned char l) : MappingId(id, l) {}
  static inline ThreadId makeId(size_t dim);
  template <unsigned char Dim>
  static ThreadId makeId(isl::id id);

  static constexpr unsigned char kMaxDim = 3;

  static inline ThreadId x();
  static inline ThreadId y();
  static inline ThreadId z();
  /*
   * Returns the size of the mapping or Mapping::unmapped if not mapped.
   * This uses traditional internal assumptions that x<=>0, y<=>1, z<=>2.
   */
  size_t mappingSize(const tc::Block& block) const;

  struct Hash {
    size_t operator()(const ThreadId& id) const {
      return isl::IslIdIslHash().operator()(id);
    }
  };
};

#define USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ) \
  using namespace tc::polyhedral::mapping;                \
  auto BX = BlockId::x();                                 \
  (void)BX;                                               \
  auto BY = BlockId::y();                                 \
  (void)BY;                                               \
  auto BZ = BlockId::z();                                 \
  (void)BZ;                                               \
  auto TX = ThreadId::x();                                \
  (void)TX;                                               \
  auto TY = ThreadId::y();                                \
  (void)TY;                                               \
  auto TZ = ThreadId::z();                                \
  (void)TZ;
} // namespace mapping
} // namespace polyhedral
} // namespace tc

#include "tc/core/polyhedral/mapping_types-inl.h"
