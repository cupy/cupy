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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <dlpack/dlpack.h>

#include "tc/core/constants.h"
#include "tc/core/halide2isl.h"
#include "tc/core/mapping_options.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/tc2halide.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

// Reduction dims must be properly ordered
using ReductionDimSet = std::set<std::string>;
class TensorReferenceGroup;

class MappedScop;

struct Scop {
 private:
  Scop() {}

 public:
  // Should be reserved for internal use and unit testing.
  static std::unique_ptr<Scop> makeScop(
      isl::ctx ctx,
      const tc2halide::HalideComponents& components);

  // Preferred points of entry, given a TC string or a treeRef,
  // Halide IR is constructed and made a member by setting halideComponents.
  // These operations are grouped and scheduled in a halide::Stmt which becomes
  // the unit from which the scop is constructed.
  static std::unique_ptr<Scop> makeScop(isl::ctx ctx, const std::string& tc);

  static std::unique_ptr<Scop> makeScop(
      isl::ctx ctx,
      const lang::TreeRef& treeRef);

  // Clone a Scop
  static std::unique_ptr<Scop> makeScop(const Scop& scop) {
    auto res = std::unique_ptr<Scop>(new Scop());
    res->globalParameterContext = scop.globalParameterContext;
    res->halide = scop.halide;
    res->reads = scop.reads;
    res->writes = scop.writes;
    res->scheduleTreeUPtr =
        detail::ScheduleTree::makeScheduleTree(*scop.scheduleTreeUPtr);
    res->treeSyncUpdateMap = scop.treeSyncUpdateMap;
    res->defaultReductionInitMap = scop.defaultReductionInitMap;
    res->groupCounts_ = scop.groupCounts_;
    res->promotedDecls_ = scop.promotedDecls_;
    res->activePromotions_ = scop.activePromotions_;
    return res;
  }

  // Intersect globalParameterContext with extraGlobalParameterContext.
  inline void intersectContext(isl::set extraGlobalParameterContext) {
    auto context = globalParameterContext & extraGlobalParameterContext;
    globalParameterContext = context;
  }

  // Specialize a Scop with extra globalParameterContext information
  // If you want to intersect the support domain with the
  // extraGlobalParameterContext then you need to do it explicitly.
  // Otherwise ambiguities will ensue.
  // TODO: this is still subject to interpretation but intersecting seems a
  // bit final here so probably we're right not doing it.
  static std::unique_ptr<Scop> makeSpecializedScop(
      const Scop& scop,
      isl::set extraGlobalParameterContext) {
    CHECK(extraGlobalParameterContext.is_subset(scop.globalParameterContext))
        << "expected extra context " << extraGlobalParameterContext
        << " to be more specialized than " << scop.globalParameterContext;
    auto res = makeScop(scop);
    res->intersectContext(extraGlobalParameterContext);
    // **WARNING** if called before scheduling, this could result in a
    // (partially) specialized schedule, i.e. force
    // strategy.proto.fix_parameters_before_scheduling to true.
    // If you want to intersect the support domain with the
    // extraGlobalParameterContext then you need to do it explicitly.
    // Note that the access relations must be intersect with the context as
    // well to obtain consistent dependences.
    // TODO: this is still subject to interpretation but intersecting seems
    // final here so probably we're right not doing it.
    // res->domain() =
    // res->domain().intersect_params(res->globalParameterContext);
    return res;
  }

  // Specialize the Scop with respect to its globalParameterContext.
  void specializeToContext() {
    domain() = domain().intersect_params(globalParameterContext);
    reads = reads.intersect_params(globalParameterContext);
    writes = writes.intersect_params(globalParameterContext);
  }

  // Returns a set that specializes (all) the scop's parameter space to the
  // integer values passed to the function.
  // WARNING: this version relies on parameter ordering, be sure you know what
  // you are doing.
  template <typename T>
  isl::set makeContext(const std::vector<T>& sizes = std::vector<T>()) const {
    auto s = domain().get_space().params();
    return makeSpecializationSet(s, sizes);
  }

  // Returns a set that specializes the (positional) scop's subset of
  // parameter space to the integer values passed to the function.
  template <typename T>
  isl::set makeContext(
      const std::unordered_map<int, T>& sizes =
          std::unordered_map<int, T>()) const {
    auto s = domain().get_space().params();
    return makeSpecializationSet(s, sizes);
  }

  // Returns a set that specializes the named scop's subset of
  // parameter space to the integer values passed to the function.
  template <typename T>
  isl::set makeContext(
      const std::unordered_map<std::string, T>& sizes =
          std::unordered_map<std::string, T>()) const {
    auto s = domain().get_space().params();
    return makeSpecializationSet(s, sizes);
  }

  // Compute the values of parameters based on the effective sizes of the
  // tensors provided as arguments and their parametric expressions stored in
  // halide ImageParams.  We only know input sizes, output sizes are inferred.
  // Result is an isl set directly usable as context.
  isl::set makeContextFromInputs(
      const std::vector<const DLTensor*>& inputs) const;

  // Fix the values of the specified parameters in the context
  // to the corresponding specified values.
  template <typename T>
  void fixParameters(const std::unordered_map<std::string, T>& sizes) {
    intersectContext(makeContext(sizes));
  }

  // Given the context set, return the list of parameter values in the same
  // order as codegen places them in the function signature, i.e. following the
  // order of scop.params.
  std::vector<long> getParameterValues(isl::set context) const;

  isl::id nextGroupIdForTensor(isl::id tensorId) {
    auto ctx = domain().get_ctx();
    std::stringstream ss;
    ss << "_" << tensorId.get_name() << "_" << groupCounts_[tensorId]++;
    return isl::id(ctx, ss.str());
  }

  // Assuming redPoint is a reduction candidate node with
  // the given reduction update statement identifier,
  // add an extension node for a reduction init and
  // a reduction update statement and insert the new
  // statements before and after (the children of) redPoint.
  // If redPoint is a sequence node, then the new node are inserted
  // inside that sequence node.  Otherwise, a new sequence node is created.
  //
  // The transformed shape is:
  //
  // *extension(     <- extension
  //   sequence(
  //     *filter()   <- red_init in new or existing sequence
  //     redPoint
  //     *filter()   <- red_update in new or existing sequence
  //   )
  // )
  //
  // This tree structure typically appears when one does not include the
  // innermost loop as part of an n-D tiling and mapping scheme but rather
  // does (n-K)D tiling and placement and then another level of placement
  // inside that.
  isl::id insertReductionSync1D(
      detail::ScheduleTree* redPoint,
      isl::id updateId);

  // Given a sequence node in the schedule tree, insert
  // synchronization before the child at position "pos".
  // If "pos" is equal to the number of children, then
  // the synchronization is added after the last child.
  void insertSync(detail::ScheduleTree* seqNode, size_t pos);

  // Insert synchronization after the given subtree,
  // creating a sequence node if needed.
  void insertSyncAfter(detail::ScheduleTree* tree) {
    insertExtensionLabelAfter(scheduleRoot(), tree, makeSyncId());
  }

  size_t reductionUID() const {
    static size_t count = 0;
    return count++;
  }
  size_t syncUID() const {
    static size_t count = 0;
    return count++;
  }

  isl::id makeSyncId() const {
    auto ctx = domain().get_ctx();
    return isl::id(ctx, std::string(kSyncIdPrefix) + std::to_string(syncUID()));
  }

  static bool isSyncId(isl::id id) {
    if (!id.has_name()) {
      return false;
    }
    auto name = id.get_name();
    if (name.find(kSyncIdPrefix) != 0) {
      return false;
    }
    name = name.substr(std::string(kSyncIdPrefix).size());
    char* end;
    std::strtol(name.c_str(), &end, 10);
    if (end - name.c_str() != name.size()) {
      return false;
    }
    return true;
  }

  static isl::id makeRefId(isl::ctx ctx) {
    static thread_local size_t count = 0;
    return isl::id(ctx, std::string("__tc_ref_") + std::to_string(count++));
  }

  std::pair<isl::id, isl::id> makeReductionSpecialIds(isl::id updateId) {
    auto uid = reductionUID();
    auto treeSyncId = isl::id(
        domain().get_ctx(), std::string("red_update") + std::to_string(uid));
    auto reductionInitId = isl::id(
        domain().get_ctx(), std::string("red_init") + std::to_string(uid));
    CHECK_EQ(0, treeSyncUpdateMap.count(treeSyncId));
    CHECK_EQ(0, defaultReductionInitMap.count(treeSyncId));

    treeSyncUpdateMap.emplace(treeSyncId, updateId);
    defaultReductionInitMap.emplace(treeSyncId, reductionInitId);
    return std::make_pair(treeSyncId, reductionInitId);
  }

  bool isTreeSyncId(isl::id id) const {
    return treeSyncUpdateMap.count(id) == 1;
  }

  bool isDefaultReductionInitId(isl::id id) const {
    for (const auto& p : defaultReductionInitMap) {
      if (p.second == id) {
        return true;
      }
    }
    return false;
  }

  isl::id getReductionUpdateForDefaultInit(isl::id id) const {
    for (const auto& p : defaultReductionInitMap) {
      if (p.second == id) {
        return treeSyncUpdateMap.at(p.first);
      }
    }
    CHECK(false) << "not found";
    return id;
  }

  bool isReductionUpdate(isl::id id) const {
    for (const auto& kvp : treeSyncUpdateMap) {
      if (id == kvp.second) {
        return true;
      }
    }
    return false;
  }

  size_t reductionUpdatePos(isl::id id) const {
    size_t pos = 0;
    CHECK(isReductionUpdate(id));
    for (const auto& kvp : treeSyncUpdateMap) {
      if (id == kvp.second) {
        return pos;
      }
      pos++;
    }
    return -1;
  }

  void promoteEverythingAt(std::vector<size_t> pos);

  struct PromotedDecl {
    isl::id tensorId;
    std::vector<size_t> sizes;
  };

  struct PromotionInfo {
    std::shared_ptr<TensorReferenceGroup> group;
    isl::union_map outerSchedule;
    isl::id groupId;
  };

  const std::unordered_map<isl::id, PromotedDecl, isl::IslIdIslHash>&
  promotedDecls() const {
    return promotedDecls_;
  }

  const std::
      unordered_map<isl::id, std::vector<PromotionInfo>, isl::IslIdIslHash>&
      activePromotions() const {
    return activePromotions_;
  }

  detail::ScheduleTree* scheduleRoot() {
    return scheduleTreeUPtr.get();
  }

  const detail::ScheduleTree* scheduleRoot() const {
    return scheduleTreeUPtr.get();
  }

  // Create a Scop scheduled with a given scheduling strategy.
  static std::unique_ptr<Scop> makeScheduled(
      const Scop& scop,
      const SchedulerOptionsView& schedulerOptions);

  // Tile the outermost band.
  // Splits the band into tile loop band and point loop band where point loops
  // have fixed trip counts specified in "tiling", and returns a pointer to the
  // tile loop band.
  detail::ScheduleTree* tileOuterBand(const TilingView& tiling);

  // Reschedule the schedule subtree rooted at "tree" with the
  // given scheduler options.
  void reschedule(
      detail::ScheduleTree* tree,
      const SchedulerOptionsView& schedulerOptions);

  // Find an input or an output argument given its name.
  // Assumes such argument exists.
  const Halide::OutputImageParam& findArgument(isl::id id) const;

  // Promote a tensor reference group to shared memory, inserting the copy
  // statements below the given node.  Inserts an Extension node below the give
  // node, unless there is already another Extension node which introduces
  // copies.  The Extension node has a unique Sequence child, whose children
  // perform copies from global memory, then main computation using the
  // original nodes, then copies back to global memory.  The caller is in
  // charge of inserting the synchronization nodes.
  //
  // Creates the promoted array declaration in the internal list.
  // If "forceLastExtentOdd" is set, the last extent in the declaration is
  // incremented if it is even.  This serves as a simple heuristic to reduce
  // shared memory bank conflicts.
  void promoteGroupToShared(
      isl::id tensorId,
      std::unique_ptr<TensorReferenceGroup>&& gr,
      detail::ScheduleTree* tree,
      const std::unordered_set<isl::id, isl::IslIdIslHash>& activeStmts,
      isl::union_map schedule,
      bool forceLastExtentOdd = false);

  // Given a tree node under which the promotion copy statements were
  // introduced, insert syncthread statements before and after the copies.
  // The tree should match the structure:
  //   any(
  //     extension(
  //       sequence(
  //         // <-- sync will be inserted here
  //         filter(any()), // filter that refers to read
  //         ...
  //         // <-- sync will be inserted here if filter above exists
  //         filter(any()), // at least one filter that does not refer to
  //         ...            // read/write
  //         // <-- sync will be inserted here if filter below exists
  //         filter(any()), // filter that refers to write
  //         ...
  //         // <-- sync will be inserted here
  //         )))
  //
  void insertSyncsAroundCopies(detail::ScheduleTree* tree);

 private:
  // Compute a schedule satisfying the given schedule constraints and
  // taking into account the scheduler options.
  // Note that some of the scheduler options have already been
  // taken into account during the construction of the schedule constraints.
  static std::unique_ptr<detail::ScheduleTree> computeSchedule(
      isl::schedule_constraints constraints,
      const SchedulerOptionsView& schedulerOptions);

 public:
  // Halide stuff
  struct {
    std::vector<Halide::Internal::Parameter> params;
    std::vector<std::string> idx, reductionIdx;
    std::vector<Halide::ImageParam> inputs;
    std::vector<Halide::OutputImageParam> outputs;
    std::vector<halide2isl::Reduction> reductions;
    std::unordered_map<isl::id, Halide::Internal::Stmt, isl::IslIdIslHash>
        statements;
    std::unordered_map<const Halide::Internal::IRNode*, isl::id> accesses;
  } halide;

  // Poyhedral IR
  //
  // The domain is collected from the root of the ScheduleTree; no redundant
  // state is kept.
  // By analogy with generalized functions, the domain is the "support" part
  // of the ScheduleTree "function".
  isl::union_set& domain();
  const isl::union_set domain() const;
  // A globalParameterContext is kept. This represents (partial)
  // parameter specialization coming from the outside.
  // This may be further specialized before codegen.
  // This globalParameterContext must not give rise to a context node in the
  // schedule tree.
  // This globalParameterContext is intersected with the domain of the
  // ScheduleTree for best possible specialization of polyhedral decisions and
  // transformations. By the analogy with generalized functions, the
  // globalParameterContext becomes part of the "support" of the ScheduleTree
  // "function".
  // This globalParameterContext lives in a parameter space.
  isl::set globalParameterContext; // TODO: not too happy about this name

  isl::union_map reads;
  isl::union_map writes;

 private:
  // By analogy with generalized functions, a ScheduleTree is a (piecewise
  // affine) function operating on a support.
  // The support is originally an isl::union_set corresponding to the union of
  // the iteration domains of the statements in the Scop.
  // The support must be the unique root node of the ScheduleTree and be of
  // type: ScheduleTreeElemDomain.
  std::unique_ptr<detail::ScheduleTree> scheduleTreeUPtr;

 public:
  // For reduction matching purposes we keep the following maps
  std::unordered_map<isl::id, isl::id, isl::IslIdIslHash> treeSyncUpdateMap;
  std::unordered_map<isl::id, isl::id, isl::IslIdIslHash>
      defaultReductionInitMap; // treeSyncId -> defaultInitId

 private:
  // Memory promotion stuff
  // tensorId -> number of mapped groups
  std::unordered_map<isl::id, size_t, isl::IslIdIslHash> groupCounts_;
  // groupId -> (tensorId, groupSizes)
  std::unordered_map<isl::id, PromotedDecl, isl::IslIdIslHash> promotedDecls_;
  // stmtId -> (group, partial schedule, groupId)
  std::unordered_map<isl::id, std::vector<PromotionInfo>, isl::IslIdIslHash>
      activePromotions_;
};

std::ostream& operator<<(std::ostream& os, const Scop&);

} // namespace polyhedral
} // namespace tc
