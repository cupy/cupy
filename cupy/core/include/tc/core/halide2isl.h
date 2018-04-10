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
#include <sstream>
#include <string>
#include <unordered_map>

#include <Halide/Halide.h>

#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/tc2halide.h"
#include "tc/external/isl.h"

namespace tc {
namespace halide2isl {

/// \file halide2isl.h
/// Helper functions that participate in translating Halide IR to ISL
///

/// Find and categorize all variables referenced in a piece of Halide IR
struct SymbolTable {
  std::vector<std::string> reductionVars, idxVars;
  std::vector<Halide::Internal::Parameter> params;
};
SymbolTable makeSymbolTable(const tc2halide::HalideComponents& components);

/// Make the space of all parameter values from the symbol table
isl::space makeParamSpace(isl::ctx ctx, const SymbolTable& symbolTable);

/// Make a constant-valued affine function over a space.
isl::aff makeIslAffFromInt(isl::space space, int64_t i);

// Make an affine function over a space from a Halide Expr. Returns a
// null isl::aff if the expression is not affine. Fails if Variable
// does not correspond to a parameter or set dimension of the space.
isl::aff makeIslAffFromExpr(isl::space space, const Halide::Expr& e);

typedef std::unordered_map<isl::id, Halide::Internal::Stmt, isl::IslIdIslHash>
    StatementMap;
typedef std::unordered_map<const Halide::Internal::IRNode*, isl::id> AccessMap;
struct ScheduleTreeAndAccesses {
  /// The schedule tree. This encodes the loop structure, but not the
  /// leaf statements. Leaf statements are replaced with IDs of the
  /// form S_N. The memory access patterns and the original statement
  /// for each leaf node is captured below.
  tc::polyhedral::ScheduleTreeUPtr tree;

  /// Union maps describing the reads and writes done. Uses the ids in
  /// the schedule tree to denote the containing Stmt, and tags each
  /// access with a unique reference id of the form __tc_ref_N.
  isl::union_map reads, writes;

  /// The correspondence between from Call and Provide nodes and the
  /// reference ids in the reads and writes maps.
  AccessMap accesses;

  /// The correspondence between leaf Stmts and the statement ids
  /// refered to above.
  StatementMap statements;
};

/// Make a schedule tree from a Halide Stmt, along with auxiliary data
/// structures describing the memory access patterns.
ScheduleTreeAndAccesses makeScheduleTree(
    isl::space paramSpace,
    const Halide::Internal::Stmt& s);

/// Enumerate all reductions in a statement, by looking for the
/// ReductionInit and ReductionUpdate markers inserted during lowering
/// (see tc2halide.h).
/// Each reduction object stores a reference to the init and
/// the update statement, although the init statement is probably
/// not strictly needed, and a list of reduction dimensions
/// in the domain of the update statement.
struct Reduction {
  Halide::Internal::Stmt init, update;
  std::vector<size_t> dims;
};
std::vector<Reduction> findReductions(const Halide::Internal::Stmt& s);

} // namespace halide2isl
} // namespace tc
