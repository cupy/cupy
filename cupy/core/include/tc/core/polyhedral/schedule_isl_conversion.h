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

#include <memory>
#include <vector>

#include <glog/logging.h>

#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_elem.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {
namespace detail {

isl::schedule toIslSchedule(const ScheduleTree* root_);

std::unique_ptr<ScheduleTree> fromIslSchedule(isl::schedule schedule);
bool validateSchedule(const ScheduleTree* st);
bool validateSchedule(isl::schedule schedule);
void checkValidIslSchedule(const detail::ScheduleTree* root_);
} // namespace detail
} // namespace polyhedral
} // namespace tc
