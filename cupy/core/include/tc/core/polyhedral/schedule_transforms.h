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
#include <string>
#include <unordered_set>
#include <vector>

#include "tc/core/polyhedral/functional.h"
#include "tc/core/polyhedral/mapping_types.h"
#include "tc/core/polyhedral/options.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {
////////////////////////////////////////////////////////////////////////////////
//                        Transformation functions, out-of-class
////////////////////////////////////////////////////////////////////////////////
// Starting from the "start" ScheduleTree, iteratively traverse the subtree
// using the "next" function and collect all nodes along the way.
// Stop when "next" returns nullptr.
// The returned vector begins with "start".
std::vector<detail::ScheduleTree*> collectScheduleTreesPath(
    std::function<detail::ScheduleTree*(detail::ScheduleTree*)> next,
    detail::ScheduleTree* start);
std::vector<const detail::ScheduleTree*> collectScheduleTreesPath(
    std::function<const detail::ScheduleTree*(const detail::ScheduleTree*)>
        next,
    const detail::ScheduleTree* start);

// Joins 2 perfectly nested bands into a single band.
// This is a structural transformation but it is not necessarily correct
// semantically. In particular, the user is responsible for setting the
// permutability of the band since it is generally required to perform
// dependence analysis to determine permutability.
// The coincident fields corresponding to members of the inner band are cleared.
detail::ScheduleTree* joinBands(
    detail::ScheduleTree* tree,
    bool permutable = false);

// Iteratively joins perfectly nested bands into a single band.
// This is a structural transformation but it is not necessarily correct
// semantically. In particular, the user is responsible for setting the
// permutability of the band since it is generally required to perform
// dependence analysis to determine permutability.
// The coincident fields corresponding to members of inner bands are cleared.
detail::ScheduleTree* joinBandsIterative(
    detail::ScheduleTree* tree,
    bool permutable = false);

// Split tree rooted under relativeRoot two nested trees, one with the first
// "pos" dimensions and one with the remaining dimensions.
// The schedules of the two bands live in anonymous spaces.
// This updates the current ScheduleTree and returns it so we can chain
// expressions.
detail::ScheduleTree* bandSplit(
    detail::ScheduleTree* relativeRoot,
    detail::ScheduleTree* tree,
    size_t pos);
// Split band rooted under relativeRoot into at most three nested band
// such that the band member at position "pos" is isolated
// into a single-member band.
// The schedules of the split bands live in anonymous spaces.
// Update the current ScheduleTree and return
// a pointer to band containing the isolated member.
detail::ScheduleTree* bandSplitOut(
    detail::ScheduleTree* relativeRoot,
    detail::ScheduleTree* tree,
    size_t pos);

// The semantics for this function is somewhat richer than the ISL C semantics.
// Since tiling is implemented as a simple band.mupa_ tranformation we can
// just complete it with 0 on the unspecified dimensions.
// This has the effect of pushing the non-tiled outer-loop inside the tile.
//   i.e. for i, j, k -> for i, j, ii, jj, k
//
// On the contrary if you want to keep the non-tiled outer-loop outside the
// tile, you can just specify tile size of 1 which, similarly to the current
// ISL behavior, will make it so.
//   i.e. for i, j, k -> for i, j, k, ii, jj, kk where range(kk)=[0, 1]
//
// This will automatically drop innermost sizes in excess of band->nMember()
//
// Modifies tree in place and returns it for call chaining purposes
//
// TODO: Support imperfectly nested tiling
detail::ScheduleTree* bandTile(
    detail::ScheduleTree* tree,
    const std::vector<size_t>& tileSizes,
    TileOptions tileOptions);

// Change the partial schedule of the band in place by multiplying it with the
// given scales.  The size of the "scales" vector must correspond to the number
// of band members.
//
// This will automatically drop innermost sizes in excess of band->nMember()
detail::ScheduleTree* bandScale(
    detail::ScheduleTree* tree,
    const std::vector<size_t>& scales);

// Map "pos"-th schedule dimension of the band node identified by "tree" to a
// _new_ parameter identified by "id" and limited by 0 <= id < extent.  The
// parameter must not be present in the space of partial schedule of "tree" and
// extent must be non-zero.  The mapping corresponds to inserting a filter
// node with condition 'dim % extent = id' where dim is "pos"-th
// schedule dimension.
//
// Returns a pointer to the updated band (below the inserted filter)
// for call chaining purposes.
template <typename MappingIdType>
detail::ScheduleTree* mapToParameterWithExtent(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    int pos,
    MappingIdType id,
    size_t extent);

// In a tree starting at a (relative) "root", insert a band node with the
// given partial schedule above the node identified by "tree".
//
// The tree is modified in place.
// Return a non-owning pointer to the inserted band node
// for call chaining purposes.
detail::ScheduleTree* insertBandAbove(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::multi_union_pw_aff mupa);

// Insert a band node with the given partial schedule below node "tree",
// which is assumed to have at most one child.
//
// The tree is modified in place.
// Return a non-owning pointer to the inserted band node
// for call chaining purposes.
detail::ScheduleTree* insertBandBelow(
    detail::ScheduleTree* tree,
    isl::multi_union_pw_aff mupa);

// Update the top-level conext node by intersecting it with "context".  The
// top-level context node must be located directly under the root of the tree.
// If there is no such node, insert one with universe context first.
void updateTopLevelContext(detail::ScheduleTree* root, isl::set context);

// In a tree starting at a (relative) "root", insert a sequence node with
// as only child the node identified by "tree".
//
// The tree is modified in place.
// Return a non-owning pointer to the inserted sequence node
// for call chaining purposes.
detail::ScheduleTree* insertSequenceAbove(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree);

// In a tree starting at a (relative) "root", insert an extension node with the
// given extension above the node identified by "tree".
//
// The tree is modified in place.
// Return a non-owning pointer to the inserted extension node
// for call chaining purposes.
detail::ScheduleTree* insertExtensionAbove(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::union_map extension);

// In a tree starting at a (relative) "root", insert a mapping filter node
// with the given filter above the node identified by "tree".
//
// The tree is modified in place.
// Return a non-owning pointer to the inserted filter node
// for call chaining purposes.
template <typename MappingIdType>
inline detail::ScheduleTree* insertMappingFilterAbove(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::union_set filter,
    const std::unordered_set<MappingIdType, typename MappingIdType::Hash>&
        mappingIds);

// Insert a mapping filter node below node "tree", which is assumed to have at
// most one child. The underlying isl::union_set filter is constructed from
// the arguments.
//
// The tree is modified in place.
template <typename MappingIdType>
inline void insertMappingFilterBelow(
    detail::ScheduleTree* tree,
    isl::union_set filter,
    const std::unordered_set<MappingIdType, typename MappingIdType::Hash>&
        mappingIds);

// Given a sequence node in the schedule tree, insert
// a zero-dimensional extension statement with the given identifier
// before the child at position "pos".
// If "pos" is equal to the number of children, then
// the statement is added after the last child.
void insertExtensionLabelAt(
    detail::ScheduleTree* root,
    detail::ScheduleTree* seqNode,
    size_t pos,
    isl::id id);

// Insert a zero-dimensional extension statement with the given identifier
// before node "tree".
// If "tree" is a sequence node or a grandchild of a sequence node,
// then the new statement is inserted in the right position
// of that sequence node.
// Otherwise, a new sequence node is inserted.
void insertExtensionLabelBefore(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::id id);

// Insert a zero-dimensional extension statement with the given identifier
// after node "tree".
// If "tree" is a sequence node or a grandchild of a sequence node,
// then the new statement is inserted in the right position
// of that sequence node.
// Otherwise, a new sequence node is inserted.
void insertExtensionLabelAfter(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::id id);

// Insert a sequence to ensure that the active domain elements
// in the given filter are executed before the other active domain elements.
void orderBefore(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::union_set filter);
// Insert a sequence to ensure that the active domain elements
// in the given filter are executed after the other active domain elements.
void orderAfter(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::union_set filter);

// Given a schedule defined by the ancestors of the given node,
// extend it to a schedule that also covers the node itself.
isl::union_map extendSchedule(
    const detail::ScheduleTree* node,
    isl::union_map schedule);

// Get the partial schedule defined by ancestors of the given node and the node
// itself.
isl::union_map partialSchedule(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node);

// Return the schedule defined by the ancestors of the given node.
isl::union_map prefixSchedule(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node);

// Return the concatenation of all outer band node partial schedules.
// If there are no outer band nodes, then return a zero-dimensional
// function on the universe domain of the schedule tree.
// Note that unlike isl_schedule_node_get_prefix_schedule_multi_union_pw_aff,
// this function does not take into account any intermediate filter nodes.
isl::multi_union_pw_aff prefixScheduleMupa(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* tree);

// Get the set of domain points active at the given node.  A domain
// point is active if it was not filtered away on the path from the
// root to the node.  The root must be a domain element, otherwise no
// elements would be considered active.
isl::union_set activeDomainPoints(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node);

// Get the set of statement identifiers whose domains have at least one active
// point at the given node, i.e. the statements that were not filtered away on
// the path from root to node.
std::unordered_set<isl::id, isl::IslIdIslHash> activeStatements(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node);

////////////////////////////////////////////////////////////////////////////////
// Experimental
////////////////////////////////////////////////////////////////////////////////
// Mapping filters are introduced one mapping dimension at a time.
// This merges consecutive filters.
detail::ScheduleTree* mergeConsecutiveMappingFilters(
    detail::ScheduleTree* root,
    detail::ScheduleTree* node);

} // namespace polyhedral
} // namespace tc

#include "tc/core/polyhedral/schedule_transforms-inl.h"
