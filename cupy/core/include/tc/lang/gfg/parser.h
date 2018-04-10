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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tc {
namespace parser {

int uid();

struct Node;

struct Edge {
  Edge(const Node& s, const std::string& trans, const Node& t)
      : source(s), transition(trans), target(t) {}

  Edge(Edge&&) = default;
  Edge(const Edge&) = delete;
  Edge& operator=(const Edge&) const = delete;

  const Node& source;
  std::string transition;
  const Node& target;
};

struct Node {
  Node(const std::string& n) : id(uid()), name(n) {}

  Node(Node&&) = default;
  Node(const Node&) = delete;
  Node& operator=(const Node&) const = delete;

  int id;
  std::string name;
  std::vector<Edge> outEdges;
};

struct GFG {
  GFG() {}
  Node& addNode(const std::string& name);
  const Edge& addEdge(Node& s, const std::string& transition, const Node& t);

  GFG(GFG&&) = default;
  GFG(const GFG&) = delete;
  GFG& operator=(const GFG&) const = delete;

  static GFG makeGFG(const std::string& grammar);

  std::vector<std::unique_ptr<Node>> nodes;

  friend std::ostream& operator<<(std::ostream& os, const GFG& g);

 private:
  std::unordered_map<std::string, int> name2NodeId;
};

std::ostream& operator<<(std::ostream& os, const Node& n);
std::ostream& operator<<(std::ostream& os, const Edge& e);
std::ostream& operator<<(std::ostream& os, const GFG& g);
}
} // ns tc::parser
