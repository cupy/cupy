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

#include <unordered_map>

namespace lang {
namespace {

std::unordered_map<std::string, int> builtin_functions({
    // from CUDA float documentation
    // WARNING: if you add a builtin function here that does not
    // propagate the type of its inputs using match_types in Sema.h
    // you need to modify Sema.h to be correct
    // see [BUILTIN TYPE MATCHING]

    // { "function_name", <num inputs> }
    {"acos", 1},
    {"acosh", 1},
    {"asin", 1},
    {"asinh", 1},
    {"atan2", 2},
    {"atan", 1},
    {"atanh", 1},
    {"cbrt", 1},
    {"ceil", 1},
    {"copysign", 2},
    {"cos", 1},
    {"cosh", 1},
    {"cospi", 1},
    {"cyl_bessel_i0", 1},
    {"cyl_bessel_i1", 1},
    {"erfc", 1},
    {"erfcinv", 1},
    {"erfcx", 1},
    {"erf", 1},
    {"erfinv", 1},
    {"exp10", 1},
    {"exp2", 1},
    {"exp", 1},
    {"expm1", 1},
    {"fabs", 1},
    {"fdim", 2},
    {"fdivide", 2},
    {"floor", 1},
    {"fma", 3},
    {"fmax", 2},
    {"fmin", 2},
    {"fmod", 2},
    {"hypot", 2},
    {"j0", 1},
    {"j1", 1},
    {"lgamma", 1},
    {"log10", 1},
    {"log1p", 1},
    {"log2", 1},
    {"logb", 1},
    {"log", 1},
    {"nextafter", 2},
    {"norm3d", 3},
    {"norm4d", 4},
    {"normcdf", 1},
    {"normcdfinv", 1},
    {"pow", 2},
    {"rcbrt", 1},
    {"remainder", 2},
    {"rhypot", 2},
    {"rnorm3d", 3},
    {"rnorm4d", 4},
    {"round", 1},
    {"rsqrt", 1},
    {"sin", 1},
    {"sinh", 1},
    {"sinpi", 1},
    {"sqrt", 1},
    {"tan", 1},
    {"tanh", 1},
    {"tgamma", 1},
    {"trunc", 1},
    {"y0", 1},
    {"y1", 1},
});

} // namespace
} // namespace lang
