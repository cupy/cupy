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
#include "tc/external/isl.h"

namespace isl {

///////////////////////////////////////////////////////////////////////////////
// Operations on isl::aff to perform arithmetic and create/combine with sets
///////////////////////////////////////////////////////////////////////////////
inline isl::aff operator*(int i, isl::aff A) {
  isl::val V(isl::val(A.get_ctx(), i));
  return A * V;
}

inline isl::aff operator*(isl::aff A, int i) {
  return i * A;
}

inline isl::aff operator*(isl::val V, isl::aff A) {
  isl::aff AV(A.get_local_space().domain(), V);
  return A.mul(AV);
}

inline isl::aff operator*(isl::aff A, isl::val V) {
  return V * A;
}

inline isl::aff operator/(isl::aff A, int i) {
  isl::ctx ctx = A.get_ctx();
  isl::aff T(isl::local_space(A.get_space().domain()), isl::val(ctx, i));
  return A.div(T);
}

inline isl::aff operator+(int i, isl::aff A) {
  isl::ctx ctx = A.get_ctx();
  return A + isl::val(ctx, i);
}

inline isl::aff operator+(isl::aff A, isl::val v) {
  isl::aff T(isl::local_space(A.get_space().domain()), v);
  return A.add(T);
}

inline isl::aff operator+(isl::val v, isl::aff A) {
  return A + v;
}

inline isl::aff operator+(isl::aff A, isl::aff B) {
  return A.add(B);
}

inline isl::aff operator+(isl::aff A, int i) {
  return i + A;
}

inline isl::aff operator-(isl::aff A, int i) {
  return A + (-i);
}

inline isl::aff operator-(int i, isl::aff A) {
  return (A + (-i)).neg();
}

inline isl::set operator>=(isl::aff_set A, isl::val v) {
  auto T = isl::aff(isl::local_space(A.aff.get_space().domain()), v);
  return A.aff.ge_set(T);
}

inline isl::set operator>=(isl::aff_set A, int i) {
  auto ctx = A.aff.get_ctx();
  return A >= isl::val(ctx, i);
}

inline isl::set operator>=(int i, isl::aff_set A) {
  return A.aff.neg() >= -i;
}

inline isl::set operator>=(isl::aff_set A, isl::aff B) {
  return A.aff.ge_set(B);
}

inline isl::set operator>=(isl::aff A, isl::aff_set B) {
  return A.ge_set(B.aff);
}

inline isl::set operator>(isl::aff_set A, int i) {
  return A >= (i + 1);
}

inline isl::set operator>(int i, isl::aff_set A) {
  return (i - 1) >= A;
}

inline isl::set operator>(isl::aff_set A, isl::aff B) {
  return A >= (B + 1);
}

inline isl::set operator>(isl::aff A, isl::aff_set B) {
  return (A - 1) >= B;
}

inline isl::set operator<=(isl::aff_set A, int i) {
  return A.aff.neg() >= -i;
}

inline isl::set operator<=(isl::aff_set A, isl::val v) {
  return A.aff.neg() >= v.neg();
}

inline isl::set operator<=(int i, isl::aff_set A) {
  return A >= i;
}

inline isl::set operator<=(isl::aff_set A, isl::aff B) {
  return A.aff.le_set(B);
}

inline isl::set operator<=(isl::aff A, isl::aff_set B) {
  return A.le_set(B.aff);
}

inline isl::set operator<(isl::aff_set A, int i) {
  return A <= i - 1;
}

inline isl::set operator<(int i, isl::aff_set A) {
  return i + 1 <= A;
}

inline isl::set operator<(isl::aff_set A, isl::val v) {
  return A <= v - 1;
}

inline isl::set operator<(isl::aff_set A, isl::aff B) {
  return A <= B - 1;
}

inline isl::set operator<(isl::aff A, isl::aff_set B) {
  return A + 1 <= B;
}

inline isl::set operator==(isl::aff_set A, int i) {
  return (A <= i) & (A >= i);
}

inline isl::set operator==(int i, isl::aff_set A) {
  return A == i;
}

inline isl::set operator==(isl::aff_set A, isl::aff B) {
  return (A <= B) & (A >= B);
}

inline isl::set operator==(isl::aff A, isl::aff_set B) {
  return (A <= B) & (A >= B);
}

inline isl::map operator>=(isl::aff_map A, isl::aff B) {
  return A > B - 1;
}

inline isl::map operator>(isl::aff_map A, isl::aff B) {
  auto pwA = isl::pw_aff(A.aff);
  auto pwB = isl::pw_aff(B);
  return pwA.gt_map(pwB);
}

inline isl::map operator<(isl::aff_map A, isl::aff B) {
  auto pwA = isl::pw_aff(A.aff);
  auto pwB = isl::pw_aff(B);
  return pwA.lt_map(pwB);
}

inline isl::map operator<=(isl::aff_map A, isl::aff B) {
  return A < B + 1;
}

///////////////////////////////////////////////////////////////////////////////
// Operations on isl::set and isl::union_set
///////////////////////////////////////////////////////////////////////////////
inline isl::set operator&(isl::set S1, isl::set S2) {
  return S1.intersect(S2);
}

inline isl::union_set operator&(isl::union_set S1, isl::set S2) {
  return S1.intersect(isl::union_set(S2));
}

inline isl::union_set operator&(isl::set S1, isl::union_set S2) {
  return S2 & S1;
}

inline isl::union_set operator&(isl::union_set S1, isl::union_set S2) {
  return S1.intersect(S2);
}

///////////////////////////////////////////////////////////////////////////////
// Operations on isl::set and isl::point
///////////////////////////////////////////////////////////////////////////////
inline isl::set operator&(isl::set S1, isl::point P2) {
  return S1.intersect(isl::set(P2));
}

inline isl::set operator&(isl::point P1, isl::set S2) {
  return S2 & P1;
}

} // namespace isl
