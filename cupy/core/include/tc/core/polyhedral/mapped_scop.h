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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tc/core/mapping_options.h"
#include "tc/core/polyhedral/mapping_types.h"
#include "tc/core/polyhedral/memory_promotion_heuristic.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/utils/dlpack.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

// Scop associated with fixed block and grid dimensions.
//
// Different branches of the schedule tree may be mapped to GPU blocks or
// threads.  The role of this class is to ensure that the number of required
// blocks and threads is consistent for the entire Scop.  It does so by
// requiring to provide grid and block configuration when constructing its
// instance.  Different parts of the schedule tree may be mapped to blocks and
// threads but the values remain those specified at construction.  If less
// blocks or threads is necessary to execute certain parts of the Scop, the
// blocks or threads dimensions will be further restricted locally in a
// specific branch of schedule tree.
//
// Two invariants must be preserved:
// 1. All paths from schedule tree root to its leaves must have exactly the
//    same number of block and thread mappings.  Code generation will fail if
//    it is not the case (TODO: automatically map to 1 thread and 1 block
//    instead).
// 2. Mapping to each block and thread must appear exactly once on each path
//    from schedule tree root to its leaves.  Mapping will fail if this
//    invariant is violated.
//
// Only const and copy accessors to the members of the original Scop are
// exposed since mapping to blocks and threads introduces schedule tree
// elements incompatible with other Scop modifications.
class MappedScop {
 private:
  MappedScop(
      std::unique_ptr<Scop>&& scop,
      ::tc::Grid grid,
      ::tc::Block block,
      uint64_t unroll_)
      : scop_(std::move(scop)),
        numBlocks(grid),
        numThreads(block),
        unroll(unroll_) {}

 public:
  static inline std::unique_ptr<MappedScop> makeOneBlockOneThread(
      std::unique_ptr<Scop>&& scop) {
    return std::unique_ptr<MappedScop>(new MappedScop(
        std::move(scop), ::tc::Grid{1, 1, 1}, ::tc::Block{1, 1, 1}, 1));
  }
  static inline std::unique_ptr<MappedScop> makeMappedScop(
      std::unique_ptr<Scop>&& scop,
      ::tc::Grid grid,
      ::tc::Block block,
      uint64_t unroll) {
    return std::unique_ptr<MappedScop>(
        new MappedScop(std::move(scop), grid, block, unroll));
  }

  // Apply the hand-written OuterBlockInnerThread mapping strategy.
  static std::unique_ptr<MappedScop> makeWithOuterBlockInnerThreadStrategy(
      std::unique_ptr<Scop>&& scopUPtr,
      const MappingOptions& mappingOptions);

  // Map a particular "pos"-th dimension in a _band_ node identified by "tree"
  // to the block or thread dimension.  Ancestors or descendants of "tree" must
  // not have a dimension already mapped to the same block or thread.
  inline detail::ScheduleTree*
  map(detail::ScheduleTree* tree, int pos, const mapping::BlockId& id) {
    return mapToParameterWithExtent(
        scop_->scheduleRoot(), tree, pos, id, id.mappingSize(numBlocks));
  }
  inline detail::ScheduleTree*
  map(detail::ScheduleTree* tree, int pos, const mapping::ThreadId& id) {
    return mapToParameterWithExtent(
        scop_->scheduleRoot(), tree, pos, id, id.mappingSize(numThreads));
  }

  // Given that "nMapped" identifiers of type "MappingTypeId" have already
  // been mapped, map the remaining ones (up to "nToMap") to zero
  // for all statement instances.
  template <typename MappingTypeId>
  void mapRemaining(detail::ScheduleTree* tree, size_t nMapped, size_t nToMap);

  // Fix the values of the specified parameters in the context
  // to the corresponding specified values.
  template <typename T>
  void fixParameters(const std::unordered_map<std::string, T>& sizes) {
    scop_->fixParameters(sizes);
  }

  // Insert a context node for the block and thread identifiers.
  void insertMappingContext();

  // Generate CUDA code at the current state of transformation provided a
  // name for the generated function.
  std::tuple<std::string, tc::Grid, tc::Block> codegen(
      const std::string& specializedName) const;

  // Accessors..
  // Const accessor to schedule of underlying Scop.
  inline const detail::ScheduleTree* schedule() const {
    return scop_->scheduleRoot();
  }
  // Reference to underlying scop, no ownership transfer intended.
  inline const Scop& scop() const {
    return *scop_;
  }
  inline Scop& scop() {
    return *scop_;
  }

 private:
  // Map "band" to block identifiers and then scale
  // the band members by "tileSizes".
  void mapToBlocksAndScaleBand(
      detail::ScheduleTree* band,
      std::vector<size_t> tileSizes);
  // Look for innermost reduction band members.
  // Store them in reductionBandUpdates_ and their parents
  // in reductionFromParent_.  Return true if any were found.
  bool detectReductions(detail::ScheduleTree* band);
  // Does separateReduction need to be called on this node?
  bool needReductionSeparation(const detail::ScheduleTree* st);
  // Return the schedule that will be used by mapInnermostBandsToThreads
  // for mapping to thread identifiers, with the last function
  // corresponding to thread identifier x.
  isl::multi_union_pw_aff reductionMapSchedule(const detail::ScheduleTree* st);
  // Separate out reductions that can be mapped to an entire block.
  // The remaining parts, if any, are no longer considered for replacement
  // by a library call.
  detail::ScheduleTree* separateReduction(detail::ScheduleTree* band);
  // Map "band" to thread identifiers, assuming "nInner" thread identifiers
  // have already been used and using as many remaining blockSizes values as
  // outer coincident dimensions,
  // unroll band members that execute at most "unroll" instances
  // (if nInner == 0) and
  // return the updated number of mapped thread identifiers.
  size_t mapToThreads(detail::ScheduleTree* band, size_t nInner);
  // Map innermost bands to thread identifiers and
  // return the number of mapped thread identifiers.
  size_t mapInnermostBandsToThreads(detail::ScheduleTree* st);

 private:
  std::unique_ptr<Scop> scop_;

 public:
  const ::tc::Grid numBlocks;
  const ::tc::Block numThreads;
  const uint64_t unroll;

  // The schedule depth that was mapped to Thread::x for specific parts of the
  // domain.
  // XXX: this is a partially redundant state as this information can
  // potentially be extracted from the schedule tree; however, until we get a
  // first-class MappingNode, it requires some dirty hacks.
  ThreadIdxxScheduleDepthState threadIdxxScheduleDepthState;

 private:
  // Information about a detected reduction that can potentially
  // be mapped to a library call.
  struct Reduction {
    Reduction(std::vector<isl::id> ids) : ids(ids), separated(false) {}
    // The statement identifiers of the reduction update statements.
    std::vector<isl::id> ids;
    // Has the reduction been separated out as a full block?
    bool separated;
  };
  // Map parent band of reduction band to the reduction band.
  // As a special case, the parent band may be missing,
  // in which case it is the reduction band that gets mapped to itself.
  std::unordered_map<const detail::ScheduleTree*, const detail::ScheduleTree*>
      reductionFromParent_;
  // Map isolated innermost reduction band members to information
  // about the detected reduction.
  std::map<const detail::ScheduleTree*, Reduction> reductionBandUpdates_;
};
} // namespace polyhedral
} // namespace tc
