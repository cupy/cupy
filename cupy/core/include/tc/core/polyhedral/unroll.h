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

namespace tc {
namespace polyhedral {

/*
 * For each descendant band node of "st" in "root" (possibly including
 * "st" itself) and for each member of the band, mark it for unrolling
 * if the total number of instances executed by the band member
 * is smaller than or equal to the unroll value (if any).
 * Only unroll values greater than 1 can have any effect.
 *
 * The number of executed instances is estimated based the number
 * of values attained by the corresponding affine function given
 * fixed values of the outer bands and members.
 * The prefix schedule is therefore computed first,
 * taking into account the actual set of statement instances and
 * the filters along the path from "root" to "st".
 */
void markUnroll(
    detail::ScheduleTree* root,
    detail::ScheduleTree* st,
    uint64_t unroll);

} // namespace polyhedral
} // namespace tc
