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
#include <vector>

#include "tc/lang/lexer.h"

namespace lang {

/// \file tree.h
/// Tree's are used to represent all forms of TC IR, pre- and post-typechecking.
/// Rather than have a full class hierarchy for all TC statements,
/// Trees are a slight variation of Lisp S-expressions.
/// for instance the expression a*b+1 is represented as:
/// (+ (* (ident a) (ident b)) (const 1))
/// Atoms like 'a', 'b', and '1' are represented by subclasses of Tree which
/// define stringValue() and doubleValue().
/// Everything else is a Compound object, which has a 'kind' that is a token
/// from Lexer.h's TokenKind enum, and contains a list of subtrees.
/// Like TokenKind single-character operators like '+' are representing using
/// the character itself, so add.kind() == '+'.
/// Compound objects are also always associated with a SourceRange for
/// reporting error message.
///
/// Memory management of trees is done using shared_ptr.

struct Tree;
using TreeRef = std::shared_ptr<Tree>;
using TreeList = std::vector<TreeRef>;

static const TreeList empty_trees = {};

struct Tree : std::enable_shared_from_this<Tree> {
  Tree(int kind_) : kind_(kind_) {}
  int kind() const {
    return kind_;
  }
  virtual bool isAtom() const {
    return true;
  }
  virtual const SourceRange& range() const {
    throw std::runtime_error("is an Atom");
  }
  virtual double doubleValue() const {
    throw std::runtime_error("not a TK_NUMBER");
  }
  virtual const std::string& stringValue() const {
    throw std::runtime_error("not a TK_STRING");
  }
  virtual bool boolValue() const {
    throw std::runtime_error("not a TK_BOOL_VALUE");
  }
  virtual const TreeList& trees() const {
    return empty_trees;
  }
  const TreeRef& tree(size_t i) const {
    return trees().at(i);
  }
  virtual TreeRef map(std::function<TreeRef(TreeRef)> fn) {
    return shared_from_this();
  }
  void expect(int k) {
    expect(k, trees().size());
  }
  void expect(int k, int numsubtrees) {
    if (kind() != k || trees().size() != numsubtrees) {
      std::stringstream ss;
      ss << "expected kind '" << kindToString(k) << "' with " << numsubtrees
         << " subtrees but found '" << kindToString(kind()) << "' with "
         << trees().size() << " subtrees.\n";
      range().highlight(ss);
      throw std::runtime_error(ss.str());
    }
  }
  int kind_;
};

struct String : public Tree {
  String(const std::string& value_) : Tree(TK_STRING), value_(value_) {}
  virtual const std::string& stringValue() const override {
    return value_;
  }
  template <typename... Args>
  static TreeRef create(Args&&... args) {
    return std::make_shared<String>(std::forward<Args>(args)...);
  }

 private:
  std::string value_;
};
struct Number : public Tree {
  Number(double value_) : Tree(TK_NUMBER), value_(value_) {}
  virtual double doubleValue() const override {
    return value_;
  }
  template <typename... Args>
  static TreeRef create(Args&&... args) {
    return std::make_shared<Number>(std::forward<Args>(args)...);
  }

 private:
  double value_;
};
struct Bool : public Tree {
  Bool(bool value_) : Tree(TK_BOOL_VALUE), value_(value_) {}
  virtual bool boolValue() const override {
    return value_;
  }
  template <typename... Args>
  static TreeRef create(Args&&... args) {
    return std::make_shared<Bool>(std::forward<Args>(args)...);
  }

 private:
  bool value_;
};

static SourceRange mergeRanges(SourceRange c, const TreeList& others) {
  for (auto t : others) {
    if (t->isAtom())
      continue;
    size_t s = std::min(c.start(), t->range().start());
    size_t e = std::max(c.end(), t->range().end());
    c = SourceRange(c.file_ptr(), s, e);
  }
  return c;
}

struct Compound : public Tree {
  Compound(int kind, const SourceRange& range_) : Tree(kind), range_(range_) {}
  Compound(int kind, const SourceRange& range_, TreeList&& trees_)
      : Tree(kind),
        range_(mergeRanges(range_, trees_)),
        trees_(std::move(trees_)) {}
  virtual const TreeList& trees() const override {
    return trees_;
  }
  static TreeRef
  create(int kind, const SourceRange& range_, TreeList&& trees_) {
    return std::make_shared<Compound>(kind, range_, std::move(trees_));
  }
  virtual bool isAtom() const override {
    return false;
  }
  virtual TreeRef map(std::function<TreeRef(TreeRef)> fn) override {
    TreeList trees_;
    for (auto& t : trees()) {
      trees_.push_back(fn(t));
    }
    return Compound::create(kind(), range(), std::move(trees_));
  }
  const SourceRange& range() const override {
    return range_;
  }

 private:
  SourceRange range_;
  TreeList trees_;
};

/// tree pretty printer
struct pretty_tree {
  pretty_tree(const TreeRef& tree, size_t col = 40) : tree(tree), col(col) {}
  const TreeRef& tree;
  size_t col;
  std::unordered_map<TreeRef, std::string> flat_strings;
  const std::string& get_flat(const TreeRef& t) {
    auto it = flat_strings.find(t);
    if (it != flat_strings.end())
      return it->second;

    std::stringstream out;
    switch (t->kind()) {
      case TK_NUMBER:
        out << t->doubleValue();
        break;
      case TK_STRING:
        out << t->stringValue();
        break;
      default:
        out << "(" << kindToString(t->kind());
        for (auto e : t->trees()) {
          out << " " << get_flat(e);
        }
        out << ")";
        break;
    }
    auto it_ = flat_strings.emplace(t, out.str());
    return it_.first->second;
  }
  void print(std::ostream& out, const TreeRef& t, int indent) {
    const std::string& s = get_flat(t);
    if (indent + s.size() < col || t->isAtom()) {
      out << s;
      return;
    }
    std::string k = kindToString(t->kind());
    out << "(" << k;
    for (auto e : t->trees()) {
      out << "\n" << std::string(indent + 2, ' ');
      print(out, e, indent + 2);
    }
    out << ")";
  }
};

static inline std::ostream& operator<<(std::ostream& out, pretty_tree t_) {
  t_.print(out, t_.tree, 0);
  return out << std::endl;
}

static inline std::ostream& operator<<(std::ostream& out, TreeRef t) {
  return out << pretty_tree(t);
}
} // namespace lang
