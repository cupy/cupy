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
#include "caffe2/core/logging.h"
#include "caffe2/core/registry.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

using ReferenceImplementation =
    std::function<void(NetDef*, const OperatorDef&)>;

/**
 * This is a simple registry for "reference implementations" of
 * operators.  The idea is that when we test tc compiled kernels,
 * it is useful to have a vanilla Caffe2 implementation which we
 * can compare the result against.
 *
 * Logically, the reference implementation registry is a mapping
 * from a tc operator to a Caffe2 NetDef which implements that
 * operator using plain Caffe2 operators.  For implementation
 * convenience, what we actually record is a function
 * (ReferenceImplementation) which knows how to take a (tc)
 * operator definition, and translate it into a sequence
 * of caffe2 operator definitions, placing those operators
 * into a pre-existing NetDef.  You can see examples of how
 * this works by grepping for TC_REFERENCE_IMPLEMENTATION.
 * In pictorial form, if you have:
 *
 *       input ---> tc op ---> output
 *                    ^_____ OperatorDef&
 *
 * then if you pass this OperatorDef& and a partially constructed
 * NetDef of the form (note that this NetDef has equivalently named
 * inputs):
 *
 *       input --X
 *
 * then ReferenceImplementation will add the reference implementation
 * for tc to your NetDef.
 *
 *       input ---> caffe2 op1 ---> caffe2 op2 ---> output
 *
 * ConvertNet takes an entire NetDef of tc operators, and replaces
 * all of them with reference implementations.  ReferenceImplementationRegistry
 * is a singleton for ease of defining implementations.
 */
class ReferenceImplementationRegistry {
 public:
  class Register {
   public:
    Register(const std::string& name, ReferenceImplementation func) {
      CAFFE_ENFORCE(!ReferenceImplementationRegistry::getMap().count(name));
      ReferenceImplementationRegistry::getMap()[name] = func;
    }
  };

  static void Append(NetDef* net, const OperatorDef& op);
  static NetDef ConvertNet(const NetDef& net);

 private:
  ReferenceImplementationRegistry() = delete;
  static CaffeMap<std::string, ReferenceImplementation>& getMap();
};

#define TC_REFERENCE_IMPLEMENTATION(name, func)              \
  static ::caffe2::ReferenceImplementationRegistry::Register \
      CAFFE_ANONYMOUS_VARIABLE(tc_ref##name)(#name, func);
} // namespace caffe2
