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

#include <string>

#include "tc/lang/tree.h"
#include "tc/lang/tree_views.h"

namespace lang {

// takes a tree after semantic analysis and create
// a canonicalized version that is agnostic to the choice of identifiers
TreeRef canonicalize(TreeRef tree) {
  struct Context {
    std::unordered_map<std::string, std::string> identMap;
    std::string rename(const std::string& name) {
      auto it = identMap.find(name);
      if (it != identMap.end()) {
        return it->second;
      }
      std::string canonicalName = "i" + std::to_string(identMap.size());
      identMap[name] = canonicalName;
      return canonicalName;
    }

    TreeRef apply(TreeRef node) {
      if (node->kind() == TK_IDENT) {
        return Ident::create(node->range(), rename(Ident(node).name()));
      }
      if (node->kind() == TK_APPLY) {
        throw ErrorReport(node)
            << "canonicalize is only valid on trees after Sema has been run "
            << "but it encountered a TK_APPLY node, which Sema removes";
      }
      return node->map([&](TreeRef ref) { return apply(ref); });
    }
  };

  Context ctx;
  return ctx.apply(tree);
}
} // namespace lang
