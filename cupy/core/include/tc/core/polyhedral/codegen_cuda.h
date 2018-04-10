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

#include <sstream>
#include <string>
#include <unordered_map>

#include "tc/core/polyhedral/mapped_scop.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

struct CodegenContext;
struct CodegenStatementContext;

namespace detail {

void emitDirectSubscripts(
    isl::pw_multi_aff subscripts,
    const CodegenStatementContext& context);

std::string toString(isl::pw_aff subscript);

isl::pw_aff makeAffFromMappedExpr(
    const Halide::Expr& expr,
    const CodegenStatementContext& context);

void emitHalideExpr(
    const Halide::Expr& e,
    const CodegenStatementContext& context);

void emitHalideExpr(
    const Halide::Expr& e,
    const CodegenStatementContext& context,
    const std::map<std::string, std::string>& substitutions);

void emitMappedSubscripts(
    const std::vector<Halide::Expr>& exprs,
    const CodegenStatementContext& context);

void emitMappedArguments(
    const std::vector<Halide::Expr>& exprs,
    const CodegenStatementContext& context);

void emitMappedTensorAccess(
    std::string name,
    const Halide::Internal::IRNode* node,
    const std::vector<Halide::Expr>& subscripts,
    const CodegenStatementContext& context);

} // namespace detail

using IteratorMapsType =
    std::unordered_map<isl::id, isl::pw_multi_aff, isl::IslIdIslHash>;

struct CodegenContext {
  CodegenContext(
      std::stringstream& ss_,
      const MappedScop& s,
      const IteratorMapsType& i)
      : ss(ss_), mappedScop(s), iteratorMaps(i) {}
  CodegenContext(const CodegenContext& c)
      : ss(c.ss), mappedScop(c.mappedScop), iteratorMaps(c.iteratorMaps) {}

  const Scop& scop() const {
    return mappedScop.scop();
  }

  std::stringstream& ss;
  const MappedScop& mappedScop;
  const IteratorMapsType& iteratorMaps;
};

struct CodegenStatementContext : CodegenContext {
  CodegenStatementContext(const CodegenContext& c, isl::id astId)
      : CodegenContext(c), astNodeId(astId) {}
  isl::pw_multi_aff iteratorMap() const {
    return this->iteratorMaps.at(astNodeId);
  }
  isl::id statementId() const {
    return this->iteratorMaps.at(astNodeId).get_tuple_id(isl::dim_type::out);
  }
  std::vector<Scop::PromotionInfo> activePromotions() const {
    auto stmtId = statementId();
    const auto& promotions = this->scop().activePromotions();
    if (promotions.count(stmtId) == 0) {
      return {};
    }
    return promotions.at(stmtId);
  }

  isl::id astNodeId;
};

std::string emitCudaKernel(
    const std::string& specializedName,
    const MappedScop& scop);

} // namespace polyhedral
} // namespace tc
