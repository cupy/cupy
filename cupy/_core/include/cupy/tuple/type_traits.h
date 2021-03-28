/*
 *  Copyright 2008-2018 NVIDIA Corporation
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


/*! \file type_traits.h
 *  \brief Temporarily define some type traits
 *         until nvcc can compile tr1::type_traits.
 */

namespace thrust
{

namespace detail
{
 /// helper classes [4.3].

template<typename T>
  struct add_const
{
  typedef T const type;
}; // end add_const

template<typename T>
  struct remove_const
{
  typedef T type;
}; // end remove_const

template<typename T>
  struct remove_const<const T>
{
  typedef T type;
}; // end remove_const

template<typename T>
  struct remove_volatile
{
  typedef T type;
}; // end remove_volatile

template<typename T>
  struct remove_volatile<volatile T>
{
  typedef T type;
}; // end remove_volatile

template<typename T>
  struct remove_cv
{
  typedef typename remove_const<typename remove_volatile<T>::type>::type type;
}; // end remove_cv

} // end detail

} // end thrust
