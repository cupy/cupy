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

namespace tc {

//
// Poor man's ScopeGuard
//
class ScopeGuard {
  std::function<void()> onExit;
  ScopeGuard() = delete;
  ScopeGuard(ScopeGuard&) = delete;
  ScopeGuard(ScopeGuard&&) = delete;
  ScopeGuard operator=(ScopeGuard&) = delete;
  ScopeGuard operator=(ScopeGuard&&) = delete;

 public:
  template <class F>
  ScopeGuard(const F& f) : onExit(f) {}
  ~ScopeGuard() noexcept(false) {
    onExit();
  }
};

} // namespace tc
