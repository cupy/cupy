/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

namespace thrust{

template <typename T>
  __device__
  inline complex<T> pow(const complex<T>& z, const complex<T> & exponent){
  return thrust::exp(thrust::log(z)*exponent);
}

/* This function should be changed as soon as FreeBSD's msun gets a cpow function */
template <>
  __device__
  inline complex<double> pow(const complex<double>& z, const complex<double> & exponent){
  return thrust::exp(thrust::log(z)*exponent);
}

template <typename T>
  __device__
  inline complex<T> pow(const complex<T>& z, const T & exponent){
  return thrust::exp(thrust::log(z)*exponent);
}

template <typename T>
  __device__
  inline complex<T> pow(const T & x, const complex<T> & exponent){
  return thrust::exp(::log(x)*exponent);
}

#if !defined _MSC_VER

template <typename T, typename U>
  __device__
  inline complex<typename _select_greater_type<T,U>::type > pow(const complex<T>& z, const complex<T>& exponent){
  typedef typename _select_greater_type<T,U>::type PromotedType;
  return thrust::exp(thrust::log(complex<PromotedType>(z))*complex<PromotedType>(exponent));
}

template <typename T, typename U>
  __device__
  inline complex<typename _select_greater_type<T,U>::type > pow(const complex<T>& z, const U& exponent){
  typedef typename _select_greater_type<T,U>::type PromotedType;
  return thrust::exp(thrust::log(complex<PromotedType>(z))*PromotedType(exponent));
}

template <typename T, typename U>
  __device__
  inline complex<typename _select_greater_type<T,U>::type > pow(const T& x, const complex<U>& exponent){
  typedef typename _select_greater_type<T,U>::type PromotedType;
  return thrust::exp(::log(PromotedType(x))*complex<PromotedType>(exponent));
}

#endif

}
