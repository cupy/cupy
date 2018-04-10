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
namespace detail {
template <typename MappingIdType>
inline ScheduleTreeUPtr ScheduleTree::makeMappingFilter(
    isl::union_set filter,
    const std::unordered_set<MappingIdType, typename MappingIdType::Hash>&
        mappingIds,
    std::vector<ScheduleTreeUPtr>&& children) {
  // slicing may happen below if not careful
  std::unordered_set<mapping::MappingId, typename mapping::MappingId::Hash> ids;
  for (auto id : mappingIds) {
    CHECK_EQ(1, mappingIds.count(id)) << "id: " << id << " mapped != 1 times";
    ids.insert(id);
  }
  isl::ctx ctx(filter.get_ctx());
  ScheduleTreeUPtr res(new ScheduleTree(ctx));
  res->elem_ = std::unique_ptr<ScheduleTreeElemMappingFilter>(
      new ScheduleTreeElemMappingFilter(filter, ids));
  res->type_ = ScheduleTreeType::MappingFilter;
  res->appendChildren(std::move(children));
  return res;
}
} // namespace detail
} // namespace polyhedral
} // namespace tc
