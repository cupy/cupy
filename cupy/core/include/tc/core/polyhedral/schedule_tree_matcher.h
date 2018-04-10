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

#include <tuple>
#include <utility>
#include <vector>

#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/scop.h"

namespace tc {
namespace polyhedral {

// Return the union of the reduction init statements as well as
// the identifiers of all reduction update statements
// that appear in "domain", assuming "domain" only contains
// reduction init and update statements.
// If "domain" contains any other statements, then return an empty vector
// of identifiers.
std::pair<isl::union_set, std::vector<isl::id>> reductionInitsUpdates(
    isl::union_set domain,
    const Scop& scop);

// Find the first band member that corresponds to a reduction.
// TODO: heuristic to choose the "best" band member in presence of multiple
// reductions.
int findFirstReductionDim(isl::multi_union_pw_aff islMupa, const Scop& scop);

} // namespace polyhedral
} // namespace tc

#include "tc/core/polyhedral/schedule_tree_matcher-inl.h"
