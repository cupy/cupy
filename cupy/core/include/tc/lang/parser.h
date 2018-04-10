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
#include "tc/lang/lexer.h"
#include "tc/lang/tree.h"
#include "tc/lang/tree_views.h"

namespace lang {

struct Parser {
  Parser(const std::string& str) : L(str), shared(sharedParserData()) {}

  TreeRef parseIdent() {
    auto t = L.expect(TK_IDENT);
    // whenever we parse something that has a TreeView type we always
    // use its create method so that the accessors and the constructor
    // of the Compound tree are in the same place.
    return Ident::create(t.range, t.text());
  }
  TreeRef parseConst() {
    auto t = L.expect(TK_NUMBER);
    auto type = (t.text().find('.') != std::string::npos ||
                 t.text().find('e') != std::string::npos)
        ? TK_FLOAT
        : TK_INT32;
    return Const::create(t.range, d(t.doubleValue()), c(type, t.range, {}));
  }
  // things like a 1.0 or a(4) that are not unary/binary expressions
  // and have higher precedence than all of them
  TreeRef parseBaseExp() {
    TreeRef prefix;
    if (L.cur().kind == TK_NUMBER) {
      prefix = parseConst();
    } else if (L.cur().kind == '(') {
      L.next();
      prefix = parseExp();
      L.expect(')');
    } else if (shared.isScalarType(L.cur().kind)) {
      // cast operation float(4 + a)
      auto type = parseScalarType();
      L.expect('(');
      auto value = parseExp();
      L.expect(')');
      return Cast::create(type->range(), value, type);
    } else {
      prefix = parseIdent();
      auto range = L.cur().range;
      if (L.cur().kind == '(') {
        prefix = Apply::create(range, prefix, parseExpList());
      } else if (L.nextIf('.')) {
        auto t = L.expect(TK_NUMBER);
        prefix = Select::create(range, prefix, d(t.doubleValue()));
      }
    }

    return prefix;
  }
  TreeRef
  parseTrinary(TreeRef cond, const SourceRange& range, int binary_prec) {
    auto true_branch = parseExp();
    L.expect(':');
    auto false_branch = parseExp(binary_prec);
    return c('?', range, {cond, true_branch, false_branch});
  }
  // parse the longest expression whose binary operators have
  // precedence strictly greater than 'precedence'
  // precedence == 0 will parse _all_ expressions
  // this is the core loop of 'top-down precedence parsing'
  TreeRef parseExp(int precedence = 0) {
    TreeRef prefix = nullptr;
    int unary_prec;
    if (shared.isUnary(L.cur().kind, &unary_prec)) {
      auto kind = L.cur().kind;
      auto pos = L.cur().range;
      L.next();
      prefix = c(kind, pos, {parseExp(unary_prec)});
    } else {
      prefix = parseBaseExp();
    }
    int binary_prec;
    while (shared.isBinary(L.cur().kind, &binary_prec)) {
      if (binary_prec <= precedence) // not allowed to parse something which is
        // not greater than 'precedenc'
        break;

      int kind = L.cur().kind;
      auto pos = L.cur().range;
      L.next();
      if (shared.isRightAssociative(kind))
        binary_prec--;

      // special case for trinary operator
      if (kind == '?') {
        prefix = parseTrinary(prefix, pos, binary_prec);
        continue;
      }

      prefix = c(kind, pos, {prefix, parseExp(binary_prec)});
    }
    return prefix;
  }
  TreeRef
  parseList(int begin, int sep, int end, std::function<TreeRef(int)> parse) {
    auto r = L.cur().range;
    L.expect(begin);
    TreeList elements;
    if (L.cur().kind != end) {
      int i = 0;
      do {
        elements.push_back(parse(i++));
      } while (L.nextIf(sep));
    }
    L.expect(end);
    return List::create(r, std::move(elements));
  }
  TreeRef parseNonEmptyList(int sep, std::function<TreeRef(int)> parse) {
    TreeList elements;
    int i = 0;
    do {
      elements.push_back(parse(i++));
    } while (L.nextIf(sep));
    auto range = elements.at(0)->range();
    return List::create(range, std::move(elements));
  }
  TreeRef parseExpList() {
    return parseList('(', ',', ')', [&](int i) { return parseExp(); });
  }
  TreeRef parseIdentList() {
    return parseList('(', ',', ')', [&](int i) { return parseIdent(); });
  }
  TreeRef parseRangeConstraint() {
    auto id = parseIdent();
    L.expect(TK_IN);
    auto l = parseExp();
    L.expect(':');
    auto r = parseExp();
    return RangeConstraint::create(id->range(), id, l, r);
  }
  TreeRef parseLetBinding() {
    auto ident = parseIdent();
    L.expect('=');
    auto exp = parseExp();
    return Let::create(ident->range(), ident, exp);
  }
  TreeRef parseWhereClause() {
    auto lookahead = L.lookahead();
    if (lookahead.kind == '=') {
      return parseLetBinding();
    } else if (lookahead.kind == TK_IN) {
      return parseRangeConstraint();
    } else {
      L.expect(TK_EXISTS);
      auto exp = parseExp();
      return Exists::create(exp->range(), {exp});
    }
  }
  TreeRef parseParam() {
    if (L.cur().kind == TK_IDENT) {
      auto ident = parseIdent();
      return Param::create(
          ident->range(), ident, c(TK_INFERRED, ident->range(), {}));
    }
    auto typ = parseType();
    auto ident = parseIdent();
    return Param::create(typ->range(), ident, typ);
  }
  TreeRef parseWhereClauses() {
    if (L.nextIf(TK_WHERE)) {
      return parseNonEmptyList(',', [&](int i) { return parseWhereClause(); });
    }
    return List::create(L.cur().range, {});
  }
  TreeRef parseEquivalent() {
    auto r = L.cur().range;
    if (L.nextIf(TK_EQUIVALENT)) {
      auto name = L.expect(TK_IDENT);
      auto accesses = parseExpList();
      return c(TK_OPTION, r, {Equivalent::create(r, name.text(), accesses)});
    }
    return c(TK_OPTION, r, {});
  }
  // =, +=, +=!, etc.
  TreeRef parseAssignment() {
    switch (L.cur().kind) {
      case TK_PLUS_EQ:
      case TK_TIMES_EQ:
      case TK_MIN_EQ:
      case TK_MAX_EQ:
      case TK_PLUS_EQ_B:
      case TK_TIMES_EQ_B:
      case TK_MIN_EQ_B:
      case TK_MAX_EQ_B:
      case '=':
        return c(L.next().kind, L.cur().range, {});
      default:
        L.reportError("a valid assignment operator");
        // unreachable, silence warnings
        return nullptr;
    }
  }
  TreeRef parseStmt() {
    auto ident = parseIdent();
    TreeRef list = parseOptionalIdentList();
    auto assign = parseAssignment();
    auto rhs = parseExp();
    TreeRef equivalent_statement = parseEquivalent();
    TreeRef range_statements = parseWhereClauses();
    TreeRef empty_reduction_variables = c(TK_LIST, ident->range(), {});
    return Comprehension::create(
        ident->range(),
        ident,
        list,
        assign,
        rhs,
        range_statements,
        equivalent_statement,
        empty_reduction_variables);
  }
  TreeRef parseScalarType() {
    if (shared.isScalarType(L.cur().kind)) {
      auto t = L.next();
      return c(t.kind, t.range, {});
    }
    L.reportError("a scalar type");
    return nullptr;
  }
  TreeRef parseOptionalIdentList() {
    TreeRef list = nullptr;
    if (L.cur().kind == '(') {
      list = parseIdentList();
    } else {
      list = List::create(L.cur().range, {});
    }
    return list;
  }
  TreeRef parseDimList() {
    return parseList('(', ',', ')', [&](int i) {
      if (L.cur().kind == TK_NUMBER) {
        return parseConst();
      } else {
        return parseIdent();
      }
    });
  }
  TreeRef parseOptionalDimList() {
    TreeRef list = nullptr;
    if (L.cur().kind == '(') {
      list = parseDimList();
    } else {
      list = List::create(L.cur().range, {});
    }
    return list;
  }
  TreeRef parseType() {
    auto st = parseScalarType();
    auto list = parseOptionalDimList();
    return TensorType::create(st->range(), st, list);
  }
  TreeRef parseFunction() {
    L.expect(TK_DEF);
    auto name = parseIdent();
    auto paramlist =
        parseList('(', ',', ')', [&](int i) { return parseParam(); });
    L.expect(TK_ARROW);
    auto retlist =
        parseList('(', ',', ')', [&](int i) { return parseParam(); });
    L.expect('{');
    auto r = L.cur().range;
    TreeList stmts;
    while (!L.nextIf('}')) {
      stmts.push_back(parseStmt());
    }
    auto stmts_list = List::create(r, std::move(stmts));
    return Def::create(name->range(), name, paramlist, retlist, stmts_list);
  }

  Lexer L;

 private:
  // short helpers to create nodes
  TreeRef d(double v) {
    return Number::create(v);
  }
  TreeRef s(const std::string& s) {
    return String::create(s);
  }
  TreeRef c(int kind, const SourceRange& range, TreeList&& trees) {
    return Compound::create(kind, range, std::move(trees));
  }
  SharedParserData& shared;
};
} // namespace lang
