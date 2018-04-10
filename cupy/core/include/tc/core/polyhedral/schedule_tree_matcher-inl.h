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

#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_elem.h"

namespace tc {
namespace polyhedral {

struct ScheduleTreeMatcher;

bool matchOne(ScheduleTreeMatcher matcher, const detail::ScheduleTree* tree);

// by-value everywhere for now, probably want something more memory-efficient
struct ScheduleTreeMatcher {
  friend bool matchOne(ScheduleTreeMatcher, const detail::ScheduleTree*);

  template <typename... Args>
  ScheduleTreeMatcher(detail::ScheduleTreeType type, Args... args)
      : type_(type),
        children_({args...}),
        propertyMatcher_([](const detail::ScheduleTree*) { return true; }),
        wildcard(false) {}

  detail::ScheduleTreeType type_;
  std::vector<ScheduleTreeMatcher> children_;
  std::function<bool(const detail::ScheduleTree*)> propertyMatcher_;
  bool wildcard;
};

// TODO: all this should probably go in a namespace
//
// This may also have a function<bool()> callback for consistency and finer
// grain control
template <typename... Args>
inline ScheduleTreeMatcher sequence(Args... children) {
  return ScheduleTreeMatcher(detail::ScheduleTreeType::Sequence, children...);
}

template <typename... Args>
inline ScheduleTreeMatcher domain(Args... children) {
  return ScheduleTreeMatcher(detail::ScheduleTreeType::Domain, children...);
}

template <typename... Args>
inline ScheduleTreeMatcher context(Args... children) {
  return ScheduleTreeMatcher(detail::ScheduleTreeType::Context, children...);
}

template <typename... Args>
inline ScheduleTreeMatcher filter(
    std::function<bool(isl::union_set)> propertyMatcher,
    Args... children) {
  ScheduleTreeMatcher m(detail::ScheduleTreeType::Filter, children...);
  m.propertyMatcher_ = [propertyMatcher](const detail::ScheduleTree* tree) {
    return propertyMatcher(
        tree->elemAs<detail::ScheduleTreeElemFilter>()->filter_);
  };
  return m;
}

template <typename... Args>
inline ScheduleTreeMatcher filter(
    std::function<bool(const detail::ScheduleTree* tree)> propertyMatcher,
    Args... children) {
  ScheduleTreeMatcher m(detail::ScheduleTreeType::Filter, children...);
  m.propertyMatcher_ = propertyMatcher;
  return m;
}

// the enable_if horror is necessary to have proper overload resolution in cases
// filter(), filter([](...){}) and filter(filter())
template <
    typename First,
    typename... Args,
    typename = typename std::enable_if<
        std::is_same<First, ScheduleTreeMatcher>::value>::type>
inline ScheduleTreeMatcher filter(First first, Args... children) {
  return ScheduleTreeMatcher(
      detail::ScheduleTreeType::Filter, first, children...);
}

inline ScheduleTreeMatcher filter() {
  return ScheduleTreeMatcher(detail::ScheduleTreeType::Filter);
}

// We could have mapping_filter restrict the property matcher but the
// extra-level of engineering sounds like a bad tradeoff, for now..
template <typename... Args>
inline ScheduleTreeMatcher mapping_filter(
    std::function<bool(isl::union_set)> propertyMatcher,
    Args... children) {
  ScheduleTreeMatcher m(detail::ScheduleTreeType::MappingFilter, children...);
  m.propertyMatcher_ = [propertyMatcher](const detail::ScheduleTree* tree) {
    return propertyMatcher(
        tree->elemAs<detail::ScheduleTreeElemMappingFilter>()->filter_);
  };
  return m;
}

template <typename... Args>
inline ScheduleTreeMatcher mapping_filter(
    std::function<bool(const detail::ScheduleTree* tree)> propertyMatcher,
    Args... children) {
  ScheduleTreeMatcher m(detail::ScheduleTreeType::MappingFilter, children...);
  m.propertyMatcher_ = propertyMatcher;
  return m;
}

// the enable_if horror is necessary to have proper overload resolution in cases
// mapping_filter(), mapping_filter([](...){}) and
// mapping_filter(mapping_filter())
template <
    typename First,
    typename... Args,
    typename = typename std::enable_if<
        std::is_same<First, ScheduleTreeMatcher>::value>::type>
inline ScheduleTreeMatcher mapping_filter(First first, Args... children) {
  return ScheduleTreeMatcher(
      detail::ScheduleTreeType::MappingFilter, first, children...);
}

inline ScheduleTreeMatcher mapping_filter() {
  return ScheduleTreeMatcher(detail::ScheduleTreeType::MappingFilter);
}

template <typename... Args>
inline ScheduleTreeMatcher band(
    std::function<bool(
        isl::multi_union_pw_aff mupa,
        bool permutable,
        std::vector<bool> coincident,
        std::vector<bool> unroll)> propertyMatcher,
    Args... children) {
  ScheduleTreeMatcher m(detail::ScheduleTreeType::Band, children...);
  m.propertyMatcher_ = [propertyMatcher](const detail::ScheduleTree* tree) {
    auto band = tree->elemAs<detail::ScheduleTreeElemBand>();
    return propertyMatcher(
        band->mupa_, band->permutable_, band->coincident_, band->unroll_);
  };
  return m;
}

template <
    typename First,
    typename... Args,
    typename = typename std::enable_if<
        std::is_same<First, ScheduleTreeMatcher>::value>::type>
inline ScheduleTreeMatcher band(First first, Args... children) {
  return ScheduleTreeMatcher(
      detail::ScheduleTreeType::Band, first, children...);
}

inline ScheduleTreeMatcher band() {
  return ScheduleTreeMatcher(detail::ScheduleTreeType::Band);
}

template <typename... Args>
inline ScheduleTreeMatcher extension(
    std::function<bool(isl::union_map)> propertyMatcher,
    Args... children) {
  ScheduleTreeMatcher m(detail::ScheduleTreeType::Extension, children...);
  m.propertyMatcher_ = [propertyMatcher](const detail::ScheduleTree* tree) {
    return propertyMatcher(
        tree->elemAs<detail::ScheduleTreeElemExtension>()->extension_);
  };
  return m;
}

template <
    typename First,
    typename... Args,
    typename = typename std::enable_if<
        std::is_same<First, ScheduleTreeMatcher>::value>::type>
inline ScheduleTreeMatcher extension(First first, Args... children) {
  return ScheduleTreeMatcher(
      detail::ScheduleTreeType::Extension, first, children...);
}

inline ScheduleTreeMatcher extension() {
  return ScheduleTreeMatcher(detail::ScheduleTreeType::Extension);
}

// Wildcard ScheduleTreeMatcher can match any 1 or more nodes.
// Examples:
//
//   * filter(
//       any()) matches a filter node with any non-empty subtree
//
//   * filter() matches a leaf filter
//
//   * filter(filter(), any()) matches any subtree whose root is a filter and
//     whose first child is a filter
inline ScheduleTreeMatcher any() {
  ScheduleTreeMatcher m(detail::ScheduleTreeType::Any);
  m.wildcard = true;
  return m;
}

inline bool matchOne(
    ScheduleTreeMatcher matcher,
    const detail::ScheduleTree* tree) {
  if (!tree) {
    return false;
  }
  if (matcher.wildcard) {
    return true;
  }
  if (matcher.type_ != tree->type_) {
    return false;
  }
  if (!matcher.propertyMatcher_(tree)) {
    return false;
  }
  // Special casing children cases to avoid accessing invalid memory
  // a. 0 children in either => the number of children need to match
  if (matcher.children_.size() == 0 || tree->numChildren() == 0) {
    if (matcher.children_.size() != tree->numChildren()) {
      return false;
    }
    return true;
  }
  // b. matcher.children do not end in wildcard then all children must match
  if (!matcher.children_.back().wildcard &&
      matcher.children_.size() != tree->numChildren()) {
    return false;
  }
  // c. whatever the case matcher cannot match if is has more children
  if (matcher.children_.size() > tree->numChildren()) {
    return false;
  }
  // No need to do a BFS here because we recurse anyway.
  // Only match up to the number of children of the matcher because:
  // 1. if matcher ends with "any", the remaining children are considered
  //    matched
  // 2. otherwise we must have the same number of children or we would have
  //    exited just above.
  // We still need to check well-formedness of the matcher (i.e. no wildcards
  // except in the last position)
  for (size_t i = 0; i < matcher.children_.size(); ++i) {
    CHECK(!matcher.children_[i].wildcard || i == matcher.children_.size() - 1)
        << "Error in matcher structure, wildcard must be the last child!";
    if (!matchOne(matcher.children_[i], tree->child({i}))) {
      return false;
    }
  }

  return true;
}

// TODO: we may need non-const versions of these to allow for modification
// after matching, the property matchers should still take const though.
//
// FIXME: we are "using namespace detail", specification below is redundant
inline std::vector<const detail::ScheduleTree*> matchDFSPreorder(
    ScheduleTreeMatcher matcher,
    const detail::ScheduleTree* tree) {
  std::vector<const detail::ScheduleTree*> res;
  for (auto t : detail::ScheduleTree::collectDFSPreorder(tree)) {
    if (matchOne(matcher, t)) {
      res.push_back(t);
    }
  }
  return res;
}

// Look for matches in arbitrary order.
inline std::vector<const detail::ScheduleTree*> match(
    ScheduleTreeMatcher matcher,
    const detail::ScheduleTree* tree) {
  return matchDFSPreorder(matcher, tree);
}

} // namespace polyhedral
} // namespace tc
