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

#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/external/isl.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tc {
namespace polyhedral {

enum class AccessType : short { Read, Write };

// A single dimension of the ScopedFootprint.
// The scope is defined by a specific position in a schedule tree (const
// ScheduleTree*), the user is responsible for maintaining the correspondance
// between schedule tree positions and footprints.
// Overapproximates one dimension by its lower bound, affine function of
// parameters and schedule dimensions visible around the scope, and by a
// constant size.
struct ScopedFootprintDim {
 public:
  ScopedFootprintDim(isl::aff lb, isl::val s) : lowerBound(lb), size(s) {}

 public:
  isl::aff lowerBound;
  isl::val size;
};

// Rectangular overapproximation of a tensor elements accessed through a single
// reference.  Each dimension is described independently.
// The scope is defined by a specific position in a schedule tree (const
// ScheduleTree*), the user is responsible for maintaining the correspondance
// between schedule tree positions and footprints.
struct ScopedFootprint : std::vector<ScopedFootprintDim> {
  isl::set footprint(isl::set domain) const;
  isl::multi_aff lowerBounds() const;
};

// Descriptor of tensor reference in a Scop.
// May be scoped to a specific position in a schedule tree, the user is
// responsible for maintaining the correspondance between schedule tree
// positions and scoped access relations.
class TensorReference {
 public:
  bool isRead() const {
    return type == AccessType::Read;
  }

  bool isWrite() const {
    return type == AccessType::Write;
  }

 public:
  // Original access relation in terms of the Scop domain.
  isl::map originalAccess;

  // Access relation in terms of partial schedule at the point where the
  // reference group is introduced in the tree.
  isl::map scopedAccess;

  // Access direction (read or write).
  AccessType type;

  // Unique identifier of a reference in the Scop.
  isl::id refId;
};

class TensorReferenceGroup;
using TensorGroupsInfo = std::vector<std::unique_ptr<TensorReferenceGroup>>;
typedef std::unordered_map<isl::id, TensorGroupsInfo, isl::IslIdIslHash>
    TensorGroups;

// A group of tensor references that must be handled together during memory
// promotion.  In particular, references that access the same tensor element,
// and at least one of them modifies it, should be placed in the shared/private
// memory together to avoid inconsistent values.
//
// Scoped to a specific position in a schedule tree, the user is responsible
// for maintaing the correspondance between schedule tree positions and scoped
// access relations of each reference as well as scoped footprints.
class TensorReferenceGroup {
 private:
  TensorReferenceGroup() {}

 public:
  static TensorGroups accessedBySubtree(
      const detail::ScheduleTree* tree,
      const Scop& scop);

  bool isReadOnly() const;

  // Sets of tensor elements accessed below the scoping point.
  isl::set writeFootprint() const;
  isl::set readFootprint() const;
  isl::set footprint() const {
    return writeFootprint().unite(readFootprint());
  }

  // Access relations in terms of partial schedule of the scoping point.
  isl::map scopedWrites() const;
  isl::map scopedReads() const;
  isl::map scopedAccesses() const {
    return scopedWrites().unite(scopedReads());
  }

  // Access relations in terms of Scop domain elements.
  // The resulting union relations have different domain spaces but identical
  // range spaces.
  isl::union_map originalWrites() const;
  isl::union_map originalReads() const;
  isl::union_map originalAccesses() const {
    return originalWrites().unite(originalReads());
  }

  // Rectangular overapproximation of the set of tensor elements accessed below
  // the scoping point.
  isl::set approximateFootprint() const {
    return approximation.footprint(scopedAccesses().domain());
  }

  isl::multi_aff promotion() const;
  isl::set promotedFootprint() const;

  std::vector<size_t> approximationSizes() const;

  std::unordered_set<isl::id, isl::IslIdIslHash> referenceIds() const;

  static std::unique_ptr<TensorReferenceGroup> makeJoint(
      std::unique_ptr<TensorReferenceGroup>&& g1,
      std::unique_ptr<TensorReferenceGroup>&& g2);
  static std::unique_ptr<TensorReferenceGroup> makeSingleton(
      isl::map originalAccess,
      isl::map scopedAccess,
      AccessType type);

 public:
  std::vector<std::unique_ptr<TensorReference>> references;
  ScopedFootprint approximation;
};

inline std::ostream& operator<<(std::ostream& os, const ScopedFootprint& fp) {
  int i = 0;
  for (const auto& f : fp) {
    if (i++ == 0) {
      os << "{\n";
    }
    os << f.lowerBound << " of size " << f.size << "\n";
  }
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorReference& tr) {
  os << ((tr.isRead()) ? "rd" : "wr") << " scopedAccess: " << tr.scopedAccess;
  ;
  return os;
}

inline std::ostream& operator<<(
    std::ostream& os,
    const TensorReferenceGroup& tg) {
  os << " with footprint BB: " << tg.approximation << " ";
  for (const auto& tr : tg.references) {
    os << *tr << " ";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorGroupsInfo& ti) {
  for (const auto& tg : ti) {
    os << *tg << " ";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorGroups& tg) {
  int i = 0;
  for (const auto& kvp : tg) {
    os << "id: " << kvp.first << "; acc: " << kvp.second;
    if (++i < tg.size()) {
      os << std::endl;
    }
  }
  return os;
}

detail::ScheduleTree* insertCopiesUnder(
    Scop& scop,
    detail::ScheduleTree* tree,
    const TensorReferenceGroup& group,
    isl::id tensorId,
    isl::id groupId = isl::id());
} // namespace polyhedral
} // namespace tc
