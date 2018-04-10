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
namespace polyhedral {
template <typename MappingIdType>
inline detail::ScheduleTree* insertMappingFilterAbove(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::union_set filter,
    const std::unordered_set<MappingIdType, typename MappingIdType::Hash>&
        mappingIds) {
  auto parent = tree->ancestor(root, 1);
  auto childPos = tree->positionInParent(parent);
  parent->insertChild(
      childPos,
      detail::ScheduleTree::makeMappingFilter(
          filter, mappingIds, parent->detachChild(childPos)));
  return parent->child({childPos});
}

template <typename MappingIdType>
inline void insertMappingFilterBelow(
    detail::ScheduleTree* tree,
    isl::union_set filter,
    const std::unordered_set<MappingIdType, typename MappingIdType::Hash>&
        mappingIds) {
  auto numChildren = tree->numChildren();
  CHECK_LE(numChildren, 1);
  tree->appendChild(detail::ScheduleTree::makeMappingFilter(
      filter, mappingIds, tree->detachChildren()));
}

template <typename MappingIdType>
inline detail::ScheduleTree* mapToParameterWithExtent(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    int pos,
    MappingIdType id,
    size_t extent) {
  auto band = tree->elemAs<detail::ScheduleTreeElemBand>();
  CHECK(band) << "expected a band, got " << *tree;
  CHECK_GE(pos, 0) << "dimension underflow";
  CHECK_LT(pos, band->nMember()) << "dimension overflow";
  CHECK_NE(extent, 0) << "NYI: mapping to 0";

  auto domain = activeDomainPoints(root, tree).universe();

  // Introduce the "mapping" parameter after checking it is not already present
  // in the schedule space.
  auto space = band->mupa_.get_space();
  int idPos = space.find_dim_by_id(isl::dim_type::param, id);
  if (idPos != -1) {
    for (auto upa : isl::MUPA(band->mupa_)) {
      for (auto pa : upa) {
        CHECK(not pa.pa.involves_dims(isl::dim_type::param, pos, 1));
      }
    }
  }

  // Create mapping filter by equating the newly introduced
  // parameter "id" to the "pos"-th schedule dimension modulo its extent.
  auto upa =
      band->mupa_.get_union_pw_aff(pos).mod_val(isl::val(tree->ctx_, extent));
  upa = upa.sub(isl::union_pw_aff::param_on_domain(domain, id));
  auto filter = upa.zero_union_set();
  return insertMappingFilterAbove<MappingIdType>(root, tree, filter, {id})
      ->child({0});
}
} // namespace polyhedral
} // namespace tc
