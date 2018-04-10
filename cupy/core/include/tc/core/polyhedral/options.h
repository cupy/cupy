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

#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

// Make these powers of 2 so that
// we can mask and add in a single argument
enum struct TileOptions : int {
  None = 0,
  ShiftPointLoops = 1,
  ScaleTileLoops = 2,
  Sentinel = 4 // Use this as next and do *= 2
};
bool operator&(TileOptions actual, TileOptions wanted);
TileOptions operator|(TileOptions actual, TileOptions wanted);

// Apply TileOptions to ISL options associated with the context
void applyTileOptions(isl::ctx& ctx, TileOptions options);

} // namespace polyhedral
} // namespace tc
