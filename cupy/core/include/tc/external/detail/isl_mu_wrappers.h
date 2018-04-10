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

#include <vector>

#include <isl/interface/isl.h>

#include "tc/external/isl.h"

namespace isl {

// Via "class-specific" get:
//   pma -> pa
//   umpa -> upa
//
// Via "class-specific" foreach:
//   pa -> foreach_piece (set, aff)
//   pma -> foreach_piece (set, ma)
//   upma -> foreach_pma (pma)
//   upa -> foreach_pa (pa)
//
// Via multi.h' get:
//   multi -> base e.g.:
//     mupa -> upa
//     mpa -> pa
//     ma -> a
//     mv -> v
//
// Via "class-specific" extract(space)
//   mupa -> mpa
//   umpa -> pma
//   upa -> pa
//
// Reverse listing:
//   mv -> v
//
//   ma -> a
//
//   pa -> (set, aff)
//
//   pma -> (set, ma) -> (set, aff)
//
//   upa : extract -> pa -> (set, aff)
//
//   mpa -> pa -> (set, aff)
//
//   upma -> upa -> pa -> (set, aff)
//       : extract -> pma -> (set, ma) -> (set, aff)
//
//   mupa -> upa -> pa -> (set, aff)
//       : extract -> mpa -> pa -> (set, aff)

// You have simple expressions, like isl_val for values and isl_aff for affine
// functions. If you want vectors instead, you do multi expressions, multi_val
// and multi_aff.
// In this particular case vector spaces have named dimensions
// and a tuple name, so there's a set space associated to each multi_* so
// multi_val is a vector in the vector space ```multi_val.get_space()``` and
// as everywhere in isl, spaces can be parametric, so you may have a
// parametric vector in a vector space that has both set and parameter
// dimensions.
//
// union is a different story, union_* are defined over different spaces
// for affine functions, it means they have different domain spaces.
//
// multi_* is a vector, we just need to differentiate vectors in different
// spaces of the same dimensionality. It is a math vector except that we don't
// have non-elementwise operations.
//
// multi_aff is a vector-valued function, but just a plain aff is a
// scalar-valued multivariate function plus parameters everywhere.

// Use the following API as follows:
//   isl::MUPA M(mupa);
//   cout << "MUPA: " << M.mupa << endl;
//   cout << "UPA: " << M[0].upa << endl;
//   cout << "PA: " << M[0][0].pa << endl;
//   cout << "PA[0] set: " << M[0][0][0].first << endl;
//   cout << "PA[0] aff: " << M[0][0][0].second << endl;
//

//
// multi_val is a vector of val in the vector space ```multi_val.get_space()```
// A val however is not attached to any (exposed) space (internally it seems
// to have one, see ```isl/isl_val_private.h```).
//
/* WARNING: this does not allow inplace modifications .. ugh */
struct MV : std::vector<isl::val> {
  explicit MV(isl::multi_val mv_) : mv(mv_) {
    this->reserve(mv.dim(isl::dim_type::set));
    for (size_t i = 0; i < mv.dim(isl::dim_type::set); ++i) {
      this->push_back(mv.get_val(i));
    }
  }
  isl::multi_val mv;
};

//
// multi_aff is a vector of aff in the vector space ```multi_aff.get_space()```
//
/* WARNING: this does not allow inplace modifications .. ugh */
struct MA : std::vector<isl::aff> {
  explicit MA(isl::multi_aff ma_) : ma(ma_) {
    this->reserve(ma.dim(isl::dim_type::set));
    for (size_t i = 0; i < ma.dim(isl::dim_type::set); ++i) {
      this->push_back(ma.get_aff(i));
    }
  }
  isl::multi_aff ma;
};

/* WARNING: this does not allow inplace modifications .. ugh */
struct PA : std::vector<std::pair<isl::set, isl::aff>> {
  explicit PA(isl::pw_aff pa_) : pa(pa_) {
    this->reserve(pa.n_piece());
    auto f = [&](isl::set s, isl::aff a) {
      this->push_back(std::make_pair(s, a));
    };
    pa.foreach_piece(f);
  }
  isl::pw_aff pa;
};

/* WARNING: this does not allow inplace modifications .. ugh */
struct PMA : std::vector<std::pair<isl::set, isl::multi_aff>> {
  explicit PMA(isl::pw_multi_aff pma_) : pma(pma_) {
    this->reserve(pma.n_piece());
    auto f = [&](isl::set s, isl::multi_aff ma) {
      this->push_back(std::make_pair(s, ma));
    };
    pma.foreach_piece(f);
  }
  isl::pw_multi_aff pma;
};

/* WARNING: this does not allow inplace modifications .. ugh */
struct UPA : std::vector<PA> {
  explicit UPA(isl::union_pw_aff upa_) : upa(upa_) {
    std::vector<PA> res;
    auto f = [&](isl::pw_aff pa) { this->push_back(PA(pa)); };
    upa.foreach_pw_aff(f);
  }
  PA extract(isl::space s) const {
    return PA(upa.extract_pw_aff(s));
  }
  isl::union_pw_aff upa;
};

struct MPA : std::vector<PA> {
  explicit MPA(isl::multi_pw_aff mpa_) : mpa(mpa_) {
    this->reserve(mpa.dim(isl::dim_type::set));
    for (size_t i = 0; i < mpa.dim(isl::dim_type::set); ++i) {
      this->push_back(PA(mpa.get_pw_aff(i)));
    }
  }
  isl::multi_pw_aff mpa;
};

/* WARNING: this does not allow inplace modifications .. ugh */
struct UPMA : std::vector<UPA> {
  explicit UPMA(isl::union_pw_multi_aff upma_) : upma(upma_) {
    this->reserve(upma.dim(isl::dim_type::set));
    for (size_t i = 0; i < upma.dim(isl::dim_type::set); ++i) {
      this->push_back(UPA(upma.get_union_pw_aff(i)));
    }
  }
  std::vector<PMA> pmas(isl::union_pw_multi_aff upma) const {
    std::vector<PMA> res;
    auto f = [&](isl::pw_multi_aff pma) { res.push_back(PMA(pma)); };
    upma.foreach_pw_multi_aff(f);
    return res;
  }
  PMA extract(isl::space s) const {
    return PMA(upma.extract_pw_multi_aff(s));
  }
  isl::union_pw_multi_aff upma;
};

/* WARNING: this does not allow inplace modifications .. ugh */
struct MUPA : std::vector<UPA> {
  explicit MUPA(isl::multi_union_pw_aff mupa_) : mupa(mupa_) {
    this->reserve(mupa.dim(isl::dim_type::set));
    for (size_t i = 0; i < mupa.dim(isl::dim_type::set); ++i) {
      this->push_back(UPA(mupa.get_union_pw_aff(i)));
    }
  }
  MPA extract(isl::space s) const {
    return MPA(mupa.extract_multi_pw_aff(s));
  }
  isl::multi_union_pw_aff mupa;
};

/* WARNING: this does not allow inplace modifications .. ugh */
struct UNION_SET : std::vector<isl::set> {
  UNION_SET(isl::union_set us_) : us(us_) {
    us_.foreach_set([&](isl::set s) { this->push_back(s); });
  }
  isl::union_set us;
};

template <typename T, isl::dim_type DT>
struct DimIds : public std::vector<isl::id> {
  DimIds(T s) {
    this->reserve(s.dim(DT));
    for (size_t i = 0; i < s.dim(DT); ++i) {
      this->push_back(s.get_dim_id(DT, i));
    }
  }
};

} // namespace isl
