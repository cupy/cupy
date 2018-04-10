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

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tc/core/polyhedral/options.h"
#include "tc/core/polyhedral/schedule_tree_elem.h"
#include "tc/core/utils/vararg.h"
#include "tc/external/isl.h"

#include "glog/logging.h"

namespace tc {
namespace polyhedral {
namespace detail {

// Internal representation of a polyhedral schedule information, wrapping a
// ScheduleTree, convertible to and from isl::schedule.
//
struct ScheduleTree;

} // namespace detail

using ScheduleTreeUPtr = std::unique_ptr<detail::ScheduleTree>;

namespace detail {

//////////////////////////////////////////////////////////////////////////////
//
// Schedule Trees
//
// Memory model: tree uniquely owns its children, user owns the root,
// traversals are non-owning.
//
// ScheduleTree is a data store of the ScheduleXYZ API.  It implements a
// mutable tree data structure, each ScheduleTree having a potentially empty
// list of children, with the following ownership semantics.  A ScheduleTree
// object owns its children.  When a child is added to or removed from the
// tree, the ownership is transferred from or to the caller.  Users of
// ScheduleTree own the root of the tree and may apply any ownership policies,
// e.g. share the ownership.  Users are guaranteed that, if they own a
// ScheduleTree, it is a root of a tree.  Ownership rules are enforced through
// unique_ptr and move semantics.  In particular, users are not expected to
// manipulate ScheduleTree objects by value.
//
// New ScheduleTree objects of various types or deep copies of the existing
// objects can be constructed using static factory functions, which transfer
// the ownership of the constructed object to the caller.  These functions
// optionally take a list of subtrees that will become children of the newly
// constructed tree, which takes ownership.
//
// Tree structure can be changed by appending, inserting, detaching or swapping
// the subtrees.  Only trees owned by the user can be attached, inserted or
// swapped with, in which case the ownership is transferred to the parent tree
// object.  The user is expected to own the root of the tree and the inserted
// tree, but not the insertion point. The ownership of the detached or swapped
// tree is transferred to the caller.
//
// ScheduleTrees are not supposed to have null children, which is checked in
// the construction/child manipulation in debug builds.
//
//
// Internal structure: single-linked tree (no parent pointer).
//
// Because the caller must own the root of the tree, it is always possible to
// find the parent or any ancestor tree by traversing the tree from the root.
// Subtrees are ordered and are identified by their position in the parent
// tree.
//
// Trees can be traversed, inspected and modified through raw non-owning
// pointers.  PreorderDFS, PostorderDFS and BFS traversals are provided.  Tree
// modification is in place and does not require the caller to own the
// ScheduleTree object.
//
// Tree modification functions are external to the ScheduleTree class and
// should only rely on the exposed API to avoid breaking the ownership and
// non-null guarantees.  For the sake of consistency, modification functions
// should take a raw pointer to the root of the tree as the first argument,
// even if they do not use it, and a raw pointer to the subtree being
// manipulated.  Transformation functions should account for the root pointer
// being relative, i.e. not being the actual root pointer owned by the caller,
// but rather some ancestor of the given node, above which the transformation
// has no effect (think of C++ standard library with begin/end iterators).
//
//
// Well-formedness guarantees: non-null subtrees.
//
// ScheduleTree does NOT impose any structure requirements on the tree, e.g.
// those of ISL.  A tree with a null child is ill-formed.
//
// Note that we do not enforce the isl schedule tree invariants [1] at the API
// level.  Instead, we provide a function checkValidIslSchedule() to verify
// whether a schedule that has a given ScheduleTree as root can be converted to
// an isl::schedule.  Note that, for compatibility reasons, the current
// implementation may also create isl::schedules that do not maintain isl
// schedule tree guarantees.  The behavior of isl calls on such schedules is
// undefined.
//
// The following isl invariants can be checked
// 1. root is domain/extension
// 2. only sequence/set have multiple children, these children are filters
// 3. nodes do not refer to parameters that were not previously introduced by a
//    context or a domain node
// 4. nodes do not refer to inactive domain points, i.e. those that were
//    filtered away (warning only)
// 5. union of filters contains all active domain elements
// 6. domain of an expansion contains all active domain elements
// 7. partial schedule of a band node is total for all active domain elements
// 8. extension nodes do not introduce any elements that are already active
//    domain elements
// 9. (not enforced)
// 10. that anchored nodes match the flattened space of the outer bands
//
// TODO(ftynse): implement bool checkValidIslSchedule() to check schedule
// structure without failing the run.
//
//
// [1] Verdoolaege, Guelton, Grosser & Cohen (2014). "Schedule trees". In
// IMPACT 2014.
//////////////////////////////////////////////////////////////////////////////
struct ScheduleTree {
  friend std::ostream& tc::polyhedral::detail::operator<<(
      std::ostream&,
      const tc::polyhedral::detail::ScheduleTree&);

 private:
  ScheduleTree() = delete;
  ScheduleTree(
      isl::ctx ctx,
      std::vector<ScheduleTreeUPtr>&& children,
      detail::ScheduleTreeType type,
      std::unique_ptr<ScheduleTreeElemBase>&& elem)
      : ctx_(ctx), type_(type), elem_(std::move(elem)) {
    appendChildren(std::move(children));
  }
  ScheduleTree(const ScheduleTree& st);

 public:
  explicit ScheduleTree(isl::ctx ctx);

  bool operator==(const ScheduleTree& other) const;
  bool operator!=(const ScheduleTree& other) const {
    return !(*this == other);
  }

  // Swap a tree with with the given tree.
  void swapChild(int pos, ScheduleTreeUPtr& swappee) {
    CHECK_GE(pos, 0) << "position out of children bounds";
    CHECK_LE(pos, children_.size()) << "position out of children bounds";
    CHECK(swappee.get()) << "Cannot swap in a null tree";
    std::swap(children_[pos], swappee);
  }

  // Child accessors (only in-place modification allowed)
  ScheduleTree* child(const std::vector<size_t>& positions);
  const ScheduleTree* child(const std::vector<size_t>& positions) const;
  size_t numChildren() const {
    return children_.size();
  };

  // Manipulators for the list of children.
  void insertChildren(size_t pos, std::vector<ScheduleTreeUPtr>&& children) {
    CHECK_GE(pos, 0) << "position out of children bounds";
    CHECK_LE(pos, children_.size()) << "position out of children bounds";
    for (const auto& c : children) {
      CHECK(c.get()) << "inserting null or moved-from child";
    }

    children_.insert(
        children_.begin() + pos,
        std::make_move_iterator(children.begin()),
        std::make_move_iterator(children.end()));
  }

  void insertChild(size_t pos, ScheduleTreeUPtr&& child) {
    // One cannot move from an initializer_list, so need an actual temporary
    // object here.
    insertChildren(pos, vectorFromArgs(std::move(child)));
  }

  void appendChildren(std::vector<ScheduleTreeUPtr>&& children) {
    insertChildren(children_.size(), std::move(children));
  }

  void appendChild(ScheduleTreeUPtr&& child) {
    insertChild(children_.size(), std::move(child));
  }

  ScheduleTreeUPtr detachChild(size_t pos) {
    CHECK_GE(pos, 0) << "position out of children bounds";
    CHECK_LT(pos, children_.size()) << "position out of children bounds";

    ScheduleTreeUPtr child = std::move(children_[pos]);
    children_.erase(children_.begin() + pos);
    return child;
  }

  std::vector<ScheduleTreeUPtr> detachChildren() {
    std::vector<ScheduleTreeUPtr> tmpChildren;
    std::swap(tmpChildren, children_);
    return tmpChildren;
  }

  std::vector<ScheduleTreeUPtr> replaceChildren(
      std::vector<ScheduleTreeUPtr>&& children) {
    auto oldChildren = detachChildren();
    appendChildren(std::move(children));
    return oldChildren;
  }

  ScheduleTreeUPtr replaceChild(size_t pos, ScheduleTreeUPtr&& child) {
    CHECK_GE(pos, 0) << "position out of children bounds";
    CHECK_LT(pos, children_.size()) << "position out of children bounds";

    ScheduleTreeUPtr oldChild = std::move(children_[pos]);
    children_[pos] = std::move(child);
    return oldChild;
  }

  // Helper to avoid calling collect + filter for this common case
  std::vector<ScheduleTree*> children() {
    std::vector<ScheduleTree*> res;
    res.reserve(children_.size());
    for (auto& p : children_) {
      res.push_back(p.get());
    }
    return res;
  };
  std::vector<const ScheduleTree*> children() const {
    std::vector<const ScheduleTree*> res;
    res.reserve(children_.size());
    for (const auto& p : children_) {
      res.push_back(p.get());
    }
    return res;
  };

  ScheduleTree* ancestor(ScheduleTree* relativeRoot, size_t generation);
  const ScheduleTree* ancestor(
      const ScheduleTree* relativeRoot,
      size_t generation) const;
  // Returns the ancestors up to relativeRoot in a vector. The first element
  // of the result is relativeRoot, the last element of the result is the
  // father of the "this" ScheduleTree.
  std::vector<ScheduleTree*> ancestors(ScheduleTree* relativeRoot);
  std::vector<const ScheduleTree*> ancestors(
      const ScheduleTree* relativeRoot) const;

  std::vector<size_t> positionRelativeTo(
      const ScheduleTree* relativeRoot) const;

  inline size_t positionInParent(const ScheduleTree* parent) const {
    auto p = positionRelativeTo(parent);
    CHECK_EQ(1, p.size()) << *parent << " is not the parent of " << *this;
    return p[0];
  }

  size_t scheduleDepth(const ScheduleTree* relativeRoot) const;

  //
  // Factory functions
  //
  static ScheduleTreeUPtr makeBand(
      isl::multi_union_pw_aff mupa,
      std::vector<ScheduleTreeUPtr>&& children = {});

  static ScheduleTreeUPtr makeDomain(
      isl::union_set domain,
      std::vector<ScheduleTreeUPtr>&& children = {});

  static ScheduleTreeUPtr makeContext(
      isl::set context,
      std::vector<ScheduleTreeUPtr>&& children = {});

  static ScheduleTreeUPtr makeFilter(
      isl::union_set filter,
      std::vector<ScheduleTreeUPtr>&& children = {});

  template <typename MappingIdType>
  static inline ScheduleTreeUPtr makeMappingFilter(
      isl::union_set filter,
      const std::unordered_set<MappingIdType, typename MappingIdType::Hash>&
          mappingIds,
      std::vector<ScheduleTreeUPtr>&& children = {});

  static ScheduleTreeUPtr makeExtension(
      isl::union_map extension,
      std::vector<ScheduleTreeUPtr>&& children = {});

  template <typename... Args>
  static ScheduleTreeUPtr makeBand(
      isl::multi_union_pw_aff mupa,
      Args&&... args) {
    return makeBand(
        mupa, vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeDomain(isl::union_set domain, Args&&... args) {
    return makeDomain(
        domain, vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeContext(isl::set context, Args&&... args) {
    return makeContext(
        context, vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeFilter(isl::union_set filter, Args&&... args) {
    return makeFilter(
        filter, vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename MappingIdType, typename... Args>
  static inline ScheduleTreeUPtr makeMappingFilter(
      isl::union_set filter,
      const std::unordered_set<MappingIdType, typename MappingIdType::Hash>&
          mappingIds,
      Args&&... args) {
    return makeMappingFilter(
        filter,
        mappingIds,
        vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeExtension(
      isl::union_map extension,
      Args&&... args) {
    return makeExtension(
        extension,
        vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeSet(Args&&... args) {
    return fromList<ScheduleTreeElemSet>(
        detail::ScheduleTreeType::Set, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeSequence(Args&&... args) {
    return fromList<ScheduleTreeElemSequence>(
        detail::ScheduleTreeType::Sequence, std::forward<Args>(args)...);
  }

  // Flatten nested nodes of the same type.
  void flattenSequenceOrSet() {
    // This should be enforced by the type system...
    CHECK(
        type_ == detail::ScheduleTreeType::Sequence ||
        type_ == detail::ScheduleTreeType::Set);

    // Iterate over the changing list of children. If a child has the same list
    // type as a parent, replace it with grandchildren and traverse them too.
    for (size_t i = 0; i < children_.size(); ++i) {
      if (children_[i]->type_ != type_) {
        continue;
      }
      auto grandChildren = children_[i]->detachChildren();
      detachChild(i);
      insertChildren(i, std::move(grandChildren));
      --i;
    }
  }

  // disallow empty lists in syntax
  template <typename T, typename Arg, typename... Args>
  static ScheduleTreeUPtr
  fromList(detail::ScheduleTreeType type, Arg&& arg, Args&&... args) {
    static_assert(
        std::is_base_of<ScheduleTreeElemBase, T>::value,
        "Can only construct elements derived from ScheduleTreeElemBase");
    static_assert(
        std::is_same<
            typename std::remove_reference<Arg>::type,
            ScheduleTreeUPtr>::value,
        "Arguments must be rvalue references to ScheduleTreeUPtr");

    auto ctx = arg->ctx_;
    std::vector<ScheduleTreeUPtr> children =
        vectorFromArgs(std::forward<Arg>(arg), std::forward<Args>(args)...);

    auto res = ScheduleTreeUPtr(new ScheduleTree(
        ctx,
        std::move(children),
        type,
        std::unique_ptr<ScheduleTreeElemBase>(new T)));

    if (type == detail::ScheduleTreeType::Sequence ||
        type == detail::ScheduleTreeType::Set) {
      res->flattenSequenceOrSet();
    }
    return res;
  }

  static ScheduleTreeUPtr makeScheduleTree(const ScheduleTree& tree) {
    return ScheduleTreeUPtr(new ScheduleTree(tree));
  }

  // Collect the nodes of "tree" in some arbitrary order.
  template <typename T>
  static std::vector<T> collect(T tree) {
    return collectDFSPreorder(tree);
  }
  // Collect the nodes of "tree" of the given type in some arbitrary order.
  template <typename T>
  static std::vector<T> collect(T tree, detail::ScheduleTreeType type) {
    return collectDFSPreorder(tree, type);
  }

  static std::vector<ScheduleTree*> collectDFSPostorder(ScheduleTree* tree);
  static std::vector<ScheduleTree*> collectDFSPreorder(ScheduleTree* tree);
  static std::vector<ScheduleTree*> collectDFSPostorder(
      ScheduleTree* tree,
      detail::ScheduleTreeType type);
  static std::vector<ScheduleTree*> collectDFSPreorder(
      ScheduleTree* tree,
      detail::ScheduleTreeType type);

  static std::vector<const ScheduleTree*> collectDFSPostorder(
      const ScheduleTree* tree);
  static std::vector<const ScheduleTree*> collectDFSPreorder(
      const ScheduleTree* tree);
  static std::vector<const ScheduleTree*> collectDFSPostorder(
      const ScheduleTree* tree,
      detail::ScheduleTreeType type);
  static std::vector<const ScheduleTree*> collectDFSPreorder(
      const ScheduleTree* tree,
      detail::ScheduleTreeType type);

  // View elem_ as the specified type.
  // Returns nullptr if this is not the proper type.
  // Inline impl for now, does not justify an extra -inl.h file
  template <typename T>
  T* elemAs() {
    const ScheduleTree* t = this;
    return const_cast<T*>(t->elemAs<const T>());
  }
  template <typename T>
  const T* elemAs() const {
    static_assert(
        std::is_base_of<ScheduleTreeElemBase, T>::value,
        "Must call with a class derived from ScheduleTreeElemBase");
    if (type_ != T::NodeType) {
      return nullptr;
    }
    return static_cast<const T*>(
        const_cast<const ScheduleTreeElemBase*>(elem_.get()));
  }

  // View elem_ as the specified type.
  // Returns nullptr if neither this type, **nor any of the derived types**,
  // are T.
  // Inline impl for now, does not justify an extra -inl.h file
  template <typename T>
  T* elemAsBase() {
    const ScheduleTree* t = this;
    return const_cast<T*>(t->elemAsBase<const T>());
  }
  template <typename T>
  const T* elemAsBase() const {
    static_assert(
        std::is_base_of<ScheduleTreeElemBase, T>::value,
        "Must call with a class derived from ScheduleTreeElemBase");
    // These T::NodeDerivedTypes are ugly, let's see if we can improve in the
    // future but if we want dynamic typing and to avoid enumerations at each
    // call site, which I claim we absolutely do, then we are not left with
    // many options.
    if (type_ != T::NodeType &&
        std::find(
            T::NodeDerivedTypes.begin(), T::NodeDerivedTypes.end(), type_) ==
            T::NodeDerivedTypes.end()) {
      return nullptr;
    }
    return static_cast<const T*>(
        const_cast<const ScheduleTreeElemBase*>(elem_.get()));
  }

  //
  // Data members
  //
 public:
  mutable isl::ctx ctx_;

 private:
  std::vector<ScheduleTreeUPtr> children_{};

 public:
  detail::ScheduleTreeType type_{detail::ScheduleTreeType::None};
  std::unique_ptr<ScheduleTreeElemBase> elem_{nullptr};
};

std::ostream& operator<<(std::ostream& os, const ScheduleTree& tree);

} // namespace detail

isl::union_set activeDomainPoints(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node);
} // namespace polyhedral
} // namespace tc

#include "tc/core/polyhedral/schedule_tree-inl.h"
