/*
* Copyright 1993-2024 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

/**
* \defgroup CUDA_MATH_INTRINSIC_HALF Half Precision Intrinsics
* This section describes half precision intrinsic functions.
* To use these functions, include the header file \p cuda_fp16.h in your program.
* All of the functions defined here are available in device code.
* Some of the functions are also available to host compilers, please
* refer to respective functions' documentation for details.
*
* NOTE: Aggressive floating-point optimizations performed by host or device
* compilers may affect numeric behavior of the functions implemented in this
* header.
*
* The following macros are available to help users selectively enable/disable
* various definitions present in the header file:
* - \p CUDA_NO_HALF - If defined, this macro will prevent the definition of
* additional type aliases in the global namespace, helping to avoid potential
* conflicts with symbols defined in the user program.
* - \p __CUDA_NO_HALF_CONVERSIONS__ - If defined, this macro will prevent the
* use of the C++ type conversions (converting constructors and conversion
* operators) that are common for built-in floating-point types, but may be
* undesirable for \p half which is essentially a user-defined type.
* - \p __CUDA_NO_HALF_OPERATORS__ and \p __CUDA_NO_HALF2_OPERATORS__ - If
* defined, these macros will prevent the inadvertent use of usual arithmetic
* and comparison operators. This enforces the storage-only type semantics and
* prevents C++ style computations on \p half and \p half2 types.
*/

/**
* \defgroup CUDA_MATH_INTRINSIC_HALF_CONSTANTS Half Arithmetic Constants
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these constants, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF_ARITHMETIC Half Arithmetic Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF2_ARITHMETIC Half2 Arithmetic Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF_COMPARISON Half Comparison Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF2_COMPARISON Half2 Comparison Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF_MISC Half Precision Conversion and Data Movement
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF_FUNCTIONS Half Math Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF2_FUNCTIONS Half2 Math Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

#ifndef __CUDA_FP16_H__
#define __CUDA_FP16_H__

// implicitly provided by NVRTC
#if !defined(__CUDACC_RTC__)
/* bring in float2, double4, etc vector types */
#include "vector_types.h"
/* bring in operations on vector types like: make_float2 */
#include "vector_functions.h"
#endif  /* !defined(__CUDACC_RTC__) */

#define ___CUDA_FP16_STRINGIFY_INNERMOST(x) #x
#define __CUDA_FP16_STRINGIFY(x) ___CUDA_FP16_STRINGIFY_INNERMOST(x)

#if defined(__cplusplus)

/* Set up function decorations */
#if (defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3))))
#define __CUDA_FP16_DECL__ __device__
#define __CUDA_HOSTDEVICE_FP16_DECL__ __device__
#define __CUDA_HOSTDEVICE__ __device__
#elif defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define __CUDA_FP16_DECL__ static __device__ __inline__
#define __CUDA_HOSTDEVICE_FP16_DECL__ static __host__ __device__ __inline__
#define __CUDA_HOSTDEVICE__ __host__ __device__
#else /* !defined(__CUDACC__) */
#if defined(__GNUC__)
#define __CUDA_HOSTDEVICE_FP16_DECL__ static __attribute__ ((unused))
#else
#define __CUDA_HOSTDEVICE_FP16_DECL__ static
#endif /* defined(__GNUC__) */
#define __CUDA_HOSTDEVICE__
#endif /* (defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3)))) */

#define __CUDA_FP16_TYPES_EXIST__

/* Macros to allow half & half2 to be used by inline assembly */
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))

/* Forward-declaration of structures defined in "cuda_fp16.hpp" */
struct __half;
struct __half2;

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts double number to half precision in round-to-nearest-even mode
* and returns \p half with converted value.
*
* \details Converts double number \p a to half precision in round-to-nearest-even mode.
* \param[in] a - double. Is only being read.
* \returns half
* - \p a converted to half precision using round-to-nearest-even mode.
* - __double2half \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __double2half \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __double2half(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __double2half(const double a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-to-nearest-even mode
* and returns \p half with converted value. 
* 
* \details Converts float number \p a to half precision in round-to-nearest-even mode. 
* \param[in] a - float. Is only being read. 
* \returns half
* - \p a converted to half precision using round-to-nearest-even mode.
* 
* \see __float2half_rn(float) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-to-nearest-even mode
* and returns \p half with converted value.
*
* \details Converts float number \p a to half precision in round-to-nearest-even mode.
* \param[in] a - float. Is only being read. 
* \returns half
* - \p a converted to half precision using round-to-nearest-even mode.
* - __float2half_rn \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __float2half_rn \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __float2half_rn(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rn(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-towards-zero mode
* and returns \p half with converted value.
* 
* \details Converts float number \p a to half precision in round-towards-zero mode.
* \param[in] a - float. Is only being read. 
* \returns half
* - \p a converted to half precision using round-towards-zero mode.
* - __float2half_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __float2half_rz \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __float2half_rz(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rz(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-down mode
* and returns \p half with converted value.
* 
* \details Converts float number \p a to half precision in round-down mode.
* \param[in] a - float. Is only being read. 
* 
* \returns half
* - \p a converted to half precision using round-down mode.
* - __float2half_rd \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __float2half_rd \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __float2half_rd(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rd(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-up mode
* and returns \p half with converted value.
* 
* \details Converts float number \p a to half precision in round-up mode.
* \param[in] a - float. Is only being read. 
* 
* \returns half
* - \p a converted to half precision using round-up mode.
* - __float2half_ru \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __float2half_ru \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __float2half_ru(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_ru(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts \p half number to float.
* 
* \details Converts half number \p a to float.
* \param[in] a - float. Is only being read. 
* 
* \returns float
* - \p a converted to float. 
* - __half2float \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __half2float \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __half2float(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ float __half2float(const __half a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts input to half precision in round-to-nearest-even mode and
* populates both halves of \p half2 with converted value.
*
* \details Converts input \p a to half precision in round-to-nearest-even mode and
* populates both halves of \p half2 with converted value.
* \param[in] a - float. Is only being read. 
*
* \returns half2
* - The \p half2 value with both halves equal to the converted half
* precision number.
* 
* \see __float2half_rn(float) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __float2half2_rn(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts both input floats to half precision in round-to-nearest-even
* mode and returns \p half2 with converted values.
*
* \details Converts both input floats to half precision in round-to-nearest-even mode
* and combines the results into one \p half2 number. Low 16 bits of the return
* value correspond to the input \p a, high 16 bits correspond to the input \p
* b.
* \param[in] a - float. Is only being read. 
* \param[in] b - float. Is only being read. 
* 
* \returns half2
* - The \p half2 value with corresponding halves equal to the
* converted input floats.
* 
* \see __float2half_rn(float) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __floats2half2_rn(const float a, const float b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts low 16 bits of \p half2 to float and returns the result
* 
* \details Converts low 16 bits of \p half2 input \p a to 32-bit floating-point number
* and returns the result.
* \param[in] a - half2. Is only being read. 
* 
* \returns float
* - The low 16 bits of \p a converted to float.
* 
* \see __half2float(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ float __low2float(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts high 16 bits of \p half2 to float and returns the result
* 
* \details Converts high 16 bits of \p half2 input \p a to 32-bit floating-point number
* and returns the result.
* \param[in] a - half2. Is only being read. 
* 
* \returns float
* - The high 16 bits of \p a converted to float.
* 
* \see __half2float(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ float __high2float(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed char in round-towards-zero mode.
*
* \details Convert the half-precision floating-point value \p h to a signed char
* integer in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read.
*
* \returns signed char
* - \p h converted to a signed char using round-towards-zero mode.
* - __half2char_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2char_rz \cuda_math_formula (x), x > 127\end_cuda_math_formula returns SCHAR_MAX = \p 0x7F.
* - __half2char_rz \cuda_math_formula (x), x < -128\end_cuda_math_formula returns SCHAR_MIN = \p 0x80.
* - __half2char_rz(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ signed char __half2char_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned char in round-towards-zero
* mode.
*
* \details Convert the half-precision floating-point value \p h to an unsigned
* char in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read.
*
* \returns unsigned char
* - \p h converted to an unsigned char using round-towards-zero mode.
* - __half2uchar_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2uchar_rz \cuda_math_formula (x), x > 255\end_cuda_math_formula returns UCHAR_MAX = \p 0xFF.
* - __half2uchar_rz \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2uchar_rz(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned char __half2uchar_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-towards-zero mode.
*
* \details Convert the half-precision floating-point value \p h to a signed short
* integer in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read.
*
* \returns short int
* - \p h converted to a signed short integer using round-towards-zero mode.
* - __half2short_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2short_rz \cuda_math_formula (x), x > 32767\end_cuda_math_formula returns SHRT_MAX = \p 0x7FFF.
* - __half2short_rz \cuda_math_formula (x), x < -32768\end_cuda_math_formula returns SHRT_MIN = \p 0x8000.
* - __half2short_rz(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ short int __half2short_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-towards-zero
* mode.
*
* \details Convert the half-precision floating-point value \p h to an unsigned short
* integer in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read.
*
* \returns unsigned short int
* - \p h converted to an unsigned short integer using round-towards-zero mode.
* - __half2ushort_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ushort_rz \cuda_math_formula (+\infty)\end_cuda_math_formula returns USHRT_MAX = \p 0xFFFF.
* - __half2ushort_rz \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2ushort_rz(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned short int __half2ushort_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-towards-zero mode.
*
* \details Convert the half-precision floating-point value \p h to a signed integer in
* round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read.
*
* \returns int
* - \p h converted to a signed integer using round-towards-zero mode.
* - __half2int_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2int_rz \cuda_math_formula (+\infty)\end_cuda_math_formula returns INT_MAX = \p 0x7FFFFFFF.
* - __half2int_rz \cuda_math_formula (-\infty)\end_cuda_math_formula returns INT_MIN = \p 0x80000000.
* - __half2int_rz(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ int __half2int_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-towards-zero mode.
*
* \details Convert the half-precision floating-point value \p h to an unsigned integer
* in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read.
*
* \returns unsigned int
* - \p h converted to an unsigned integer using round-towards-zero mode.
* - __half2uint_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2uint_rz \cuda_math_formula (+\infty)\end_cuda_math_formula returns UINT_MAX = \p 0xFFFFFFFF.
* - __half2uint_rz \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2uint_rz(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __half2uint_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-towards-zero mode.
*
* \details Convert the half-precision floating-point value \p h to a signed 64-bit
* integer in round-towards-zero mode. NaN inputs return a long long int with hex value of \p 0x8000000000000000.
* \param[in] h - half. Is only being read.
*
* \returns long long int
* - \p h converted to a signed 64-bit integer using round-towards-zero mode.
* - __half2ll_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ll_rz \cuda_math_formula (+\infty)\end_cuda_math_formula returns LLONG_MAX = \p 0x7FFFFFFFFFFFFFFF.
* - __half2ll_rz \cuda_math_formula (-\infty)\end_cuda_math_formula returns LLONG_MIN = \p 0x8000000000000000.
* - __half2ll_rz(NaN) returns \p 0x8000000000000000.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ long long int __half2ll_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-towards-zero
* mode.
*
* \details Convert the half-precision floating-point value \p h to an unsigned 64-bit
* integer in round-towards-zero mode. NaN inputs return \p 0x8000000000000000.
* \param[in] h - half. Is only being read.
*
* \returns unsigned long long int
* - \p h converted to an unsigned 64-bit integer using round-towards-zero mode.
* - __half2ull_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ull_rz \cuda_math_formula (+\infty)\end_cuda_math_formula returns ULLONG_MAX = \p 0xFFFFFFFFFFFFFFFF.
* - __half2ull_rz \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2ull_rz(NaN) returns \p 0x8000000000000000.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned long long int __half2ull_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Vector function, combines two \p __half numbers into one \p __half2 number.
* 
* \details Combines two input \p __half number \p x and \p y into one \p __half2 number.
* Input \p x is stored in low 16 bits of the return value, input \p y is stored
* in high 16 bits of the return value.
* \param[in] x - half. Is only being read. 
* \param[in] y - half. Is only being read. 
* 
* \returns __half2
* - The \p __half2 vector with one half equal to \p x and the other to \p y. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 make_half2(const __half x, const __half y);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts both components of \p float2 number to half precision in
* round-to-nearest-even mode and returns \p half2 with converted values.
* 
* \details Converts both components of \p float2 to half precision in round-to-nearest-even
* mode and combines the results into one \p half2 number. Low 16 bits of the
* return value correspond to \p a.x and high 16 bits of the return value
* correspond to \p a.y.
* \param[in] a - float2. Is only being read. 
*  
* \returns half2
* - The \p half2 which has corresponding halves equal to the
* converted \p float2 components.
* 
* \see __float2half_rn(float) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __float22half2_rn(const float2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts both halves of \p half2 to \p float2 and returns the result.
* 
* \details Converts both halves of \p half2 input \p a to \p float2 and returns the
* result.
* \param[in] a - half2. Is only being read. 
* 
* \returns float2
* - \p a converted to \p float2.
* 
* \see __half2float(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ float2 __half22float2(const __half2 a);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-to-nearest-even mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed integer in
* round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns int
* - \p h converted to a signed integer using round-to-nearest-even mode.
* - __half2int_rn \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2int_rn \cuda_math_formula (+\infty)\end_cuda_math_formula returns INT_MAX = \p 0x7FFFFFFF.
* - __half2int_rn \cuda_math_formula (-\infty)\end_cuda_math_formula returns INT_MIN = \p 0x80000000.
* - __half2int_rn(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ int __half2int_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-down mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed integer in
* round-down mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns int
* - \p h converted to a signed integer using round-down mode.
* - __half2int_rd \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2int_rd \cuda_math_formula (+\infty)\end_cuda_math_formula returns INT_MAX = \p 0x7FFFFFFF.
* - __half2int_rd \cuda_math_formula (-\infty)\end_cuda_math_formula returns INT_MIN = \p 0x80000000.
* - __half2int_rd(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ int __half2int_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-up mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed integer in
* round-up mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns int
* - \p h converted to a signed integer using round-up mode.
* - __half2int_ru \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2int_ru \cuda_math_formula (+\infty)\end_cuda_math_formula returns INT_MAX = \p 0x7FFFFFFF.
* - __half2int_ru \cuda_math_formula (-\infty)\end_cuda_math_formula returns INT_MIN = \p 0x80000000.
* - __half2int_ru(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ int __half2int_ru(const __half h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-to-nearest-even mode.
* 
* \details Convert the signed integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __int2half_rn(const int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-towards-zero mode.
* 
* \details Convert the signed integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __int2half_rz(const int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-down mode.
* 
* \details Convert the signed integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __int2half_rd(const int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-up mode.
* 
* \details Convert the signed integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __int2half_ru(const int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-to-nearest-even
* mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed short
* integer in round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns short int
* - \p h converted to a signed short integer using round-to-nearest-even mode.
* - __half2short_rn \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2short_rn \cuda_math_formula (x), x > 32767\end_cuda_math_formula returns SHRT_MAX = \p 0x7FFF.
* - __half2short_rn \cuda_math_formula (x), x < -32768\end_cuda_math_formula returns SHRT_MIN = \p 0x8000.
* - __half2short_rn(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ short int __half2short_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-down mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed short
* integer in round-down mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns short int
* - \p h converted to a signed short integer using round-down mode.
* - __half2short_rd \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2short_rd \cuda_math_formula (x), x > 32767\end_cuda_math_formula returns SHRT_MAX = \p 0x7FFF.
* - __half2short_rd \cuda_math_formula (x), x < -32768\end_cuda_math_formula returns SHRT_MIN = \p 0x8000.
* - __half2short_rd(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ short int __half2short_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-up mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed short
* integer in round-up mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns short int
* - \p h converted to a signed short integer using round-up mode.
* - __half2short_ru \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2short_ru \cuda_math_formula (x), x > 32767\end_cuda_math_formula returns SHRT_MAX = \p 0x7FFF.
* - __half2short_ru \cuda_math_formula (x), x < -32768\end_cuda_math_formula returns SHRT_MIN = \p 0x8000.
* - __half2short_ru(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ short int __half2short_ru(const __half h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-to-nearest-even
* mode.
* 
* \details Convert the signed short integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - short int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __short2half_rn(const short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-towards-zero mode.
* 
* \details Convert the signed short integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - short int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __short2half_rz(const short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-down mode.
* 
* \details Convert the signed short integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - short int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __short2half_rd(const short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-up mode.
* 
* \details Convert the signed short integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - short int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __short2half_ru(const short int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-to-nearest-even mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned integer
* in round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned int
* - \p h converted to an unsigned integer using round-to-nearest-even mode.
* - __half2uint_rn \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2uint_rn \cuda_math_formula (+\infty)\end_cuda_math_formula returns UINT_MAX = \p 0xFFFFFFFF.
* - __half2uint_rn \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2uint_rn(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned int __half2uint_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-down mode.
*
* \details Convert the half-precision floating-point value \p h to an unsigned integer
* in round-down mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
*
* \returns unsigned int
* - \p h converted to an unsigned integer using round-down mode.
* - __half2uint_rd \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2uint_rd \cuda_math_formula (+\infty)\end_cuda_math_formula returns UINT_MAX = \p 0xFFFFFFFF.
* - __half2uint_rd \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2uint_rd(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned int __half2uint_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-up mode.
*
* \details Convert the half-precision floating-point value \p h to an unsigned integer
* in round-up mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
*
* \returns unsigned int
* - \p h converted to an unsigned integer using round-up mode.
* - __half2uint_ru \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2uint_ru \cuda_math_formula (+\infty)\end_cuda_math_formula returns UINT_MAX = \p 0xFFFFFFFF.
* - __half2uint_ru \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2uint_ru(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned int __half2uint_ru(const __half h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-to-nearest-even mode.
* 
* \details Convert the unsigned integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - unsigned int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __uint2half_rn(const unsigned int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-towards-zero mode.
* 
* \details Convert the unsigned integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - unsigned int. Is only being read. 
* 
* \returns half
* - \p i converted to half.  
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __uint2half_rz(const unsigned int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-down mode.
* 
* \details Convert the unsigned integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - unsigned int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __uint2half_rd(const unsigned int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-up mode.
* 
* \details Convert the unsigned integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - unsigned int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __uint2half_ru(const unsigned int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-to-nearest-even
* mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned short
* integer in round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned short int
* - \p h converted to an unsigned short integer using round-to-nearest-even mode.
* - __half2ushort_rn \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ushort_rn \cuda_math_formula (+\infty)\end_cuda_math_formula returns USHRT_MAX = \p 0xFFFF.
* - __half2ushort_rn \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2ushort_rn(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-down mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned short
* integer in round-down mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned short int
* - \p h converted to an unsigned short integer using round-down mode.
* - __half2ushort_rd \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ushort_rd \cuda_math_formula (+\infty)\end_cuda_math_formula returns USHRT_MAX = \p 0xFFFF.
* - __half2ushort_rd \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2ushort_rd(NaN) returns 0.
*/
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-up mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned short
* integer in round-up mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned short int
* - \p h converted to an unsigned short integer using round-up mode.
* - __half2ushort_ru \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ushort_ru \cuda_math_formula (+\infty)\end_cuda_math_formula returns USHRT_MAX = \p 0xFFFF.
* - __half2ushort_ru \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2ushort_ru(NaN) returns 0.
*/
__CUDA_FP16_DECL__ unsigned short int __half2ushort_ru(const __half h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-to-nearest-even
* mode.
* 
* \details Convert the unsigned short integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - unsigned short int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort2half_rn(const unsigned short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-towards-zero
* mode.
* 
* \details Convert the unsigned short integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - unsigned short int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort2half_rz(const unsigned short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-down mode.
* 
* \details Convert the unsigned short integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - unsigned short int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort2half_rd(const unsigned short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-up mode.
* 
* \details Convert the unsigned short integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - unsigned short int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort2half_ru(const unsigned short int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-to-nearest-even
* mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned 64-bit
* integer in round-to-nearest-even mode. NaN inputs return \p 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned long long int
* - \p h converted to an unsigned 64-bit integer using round-to-nearest-even mode.
* - __half2ull_rn \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ull_rn \cuda_math_formula (+\infty)\end_cuda_math_formula returns ULLONG_MAX = \p 0xFFFFFFFFFFFFFFFF.
* - __half2ull_rn \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2ull_rn(NaN) returns \p 0x8000000000000000.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-down mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned 64-bit
* integer in round-down mode. NaN inputs return \p 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned long long int
* - \p h converted to an unsigned 64-bit integer using round-down mode.
* - __half2ull_rd \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ull_rd \cuda_math_formula (+\infty)\end_cuda_math_formula returns ULLONG_MAX = \p 0xFFFFFFFFFFFFFFFF.
* - __half2ull_rd \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2ull_rd(NaN) returns \p 0x8000000000000000.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-up mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned 64-bit
* integer in round-up mode. NaN inputs return \p 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned long long int
* - \p h converted to an unsigned 64-bit integer using round-up mode.
* - __half2ull_ru \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ull_ru \cuda_math_formula (+\infty)\end_cuda_math_formula returns ULLONG_MAX = \p 0xFFFFFFFFFFFFFFFF.
* - __half2ull_ru \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __half2ull_ru(NaN) returns \p 0x8000000000000000.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned long long int __half2ull_ru(const __half h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-to-nearest-even
* mode.
* 
* \details Convert the unsigned 64-bit integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - unsigned long long int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ull2half_rn(const unsigned long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-towards-zero
* mode.
* 
* \details Convert the unsigned 64-bit integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - unsigned long long int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ull2half_rz(const unsigned long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-down mode.
* 
* \details Convert the unsigned 64-bit integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - unsigned long long int. Is only being read. 
* 
* \returns half
* - \p i converted to half.  
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ull2half_rd(const unsigned long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-up mode.
* 
* \details Convert the unsigned 64-bit integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - unsigned long long int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ull2half_ru(const unsigned long long int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-to-nearest-even
* mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed 64-bit
* integer in round-to-nearest-even mode. NaN inputs return a long long int with hex value of \p 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns long long int
* - \p h converted to a signed 64-bit integer using round-to-nearest-even mode.
* - __half2ll_rn \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ll_rn \cuda_math_formula (+\infty)\end_cuda_math_formula returns LLONG_MAX = \p 0x7FFFFFFFFFFFFFFF.
* - __half2ll_rn \cuda_math_formula (-\infty)\end_cuda_math_formula returns LLONG_MIN = \p 0x8000000000000000.
* - __half2ll_rn(NaN) returns \p 0x8000000000000000.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ long long int __half2ll_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-down mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed 64-bit
* integer in round-down mode. NaN inputs return a long long int with hex value of \p 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns long long int
* - \p h converted to a signed 64-bit integer using round-down mode.
* - __half2ll_rd \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ll_rd \cuda_math_formula (+\infty)\end_cuda_math_formula returns LLONG_MAX = \p 0x7FFFFFFFFFFFFFFF.
* - __half2ll_rd \cuda_math_formula (-\infty)\end_cuda_math_formula returns LLONG_MIN = \p 0x8000000000000000.
* - __half2ll_rd(NaN) returns \p 0x8000000000000000.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ long long int __half2ll_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-up mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed 64-bit
* integer in round-up mode. NaN inputs return a long long int with hex value of \p 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns long long int
* - \p h converted to a signed 64-bit integer using round-up mode.
* - __half2ll_ru \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __half2ll_ru \cuda_math_formula (+\infty)\end_cuda_math_formula returns LLONG_MAX = \p 0x7FFFFFFFFFFFFFFF.
* - __half2ll_ru \cuda_math_formula (-\infty)\end_cuda_math_formula returns LLONG_MIN = \p 0x8000000000000000.
* - __half2ll_ru(NaN) returns \p 0x8000000000000000.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ long long int __half2ll_ru(const __half h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-to-nearest-even
* mode.
* 
* \details Convert the signed 64-bit integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - long long int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ll2half_rn(const long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-towards-zero mode.
* 
* \details Convert the signed 64-bit integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - long long int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ll2half_rz(const long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-down mode.
* 
* \details Convert the signed 64-bit integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - long long int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ll2half_rd(const long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-up mode.
* 
* \details Convert the signed 64-bit integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - long long int. Is only being read. 
* 
* \returns half
* - \p i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ll2half_ru(const long long int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Truncate input argument to the integral part.
* 
* \details Round \p h to the largest integer value that does not exceed \p h in
* magnitude.
* \param[in] h - half. Is only being read. 
* 
* \returns half
* - The truncated value. 
* - htrunc(
* \cuda_math_formula \pm 0 \end_cuda_math_formula
* ) returns 
* \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - htrunc(
* \cuda_math_formula \pm \infty \end_cuda_math_formula
* ) returns 
* \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - htrunc(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half htrunc(const __half h);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculate ceiling of the input argument.
* 
* \details Compute the smallest integer value not less than \p h.
* \param[in] h - half. Is only being read. 
* 
* \returns half
* - The smallest integer value not less than \p h. 
* - hceil(
* \cuda_math_formula \pm 0 \end_cuda_math_formula
* ) returns 
* \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - hceil(
* \cuda_math_formula \pm \infty \end_cuda_math_formula
* ) returns 
* \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - hceil(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hceil(const __half h);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculate the largest integer less than or equal to \p h.
* 
* \details Calculate the largest integer value which is less than or equal to \p h.
* \param[in] h - half. Is only being read. 
* 
* \returns half
* - The largest integer value which is less than or equal to \p h. 
* - hfloor(
* \cuda_math_formula \pm 0 \end_cuda_math_formula
* ) returns 
* \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - hfloor(
* \cuda_math_formula \pm \infty \end_cuda_math_formula
* ) returns 
* \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - hfloor(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hfloor(const __half h);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Round input to nearest integer value in half-precision floating-point
* number.
* 
* \details Round \p h to the nearest integer value in half-precision floating-point
* format, with halfway cases rounded to the nearest even integer value.
* \param[in] h - half. Is only being read. 
* 
* \returns half
* - The nearest integer to \p h. 
* - hrint(
* \cuda_math_formula \pm 0 \end_cuda_math_formula
* ) returns 
* \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - hrint(
* \cuda_math_formula \pm \infty \end_cuda_math_formula
* ) returns 
* \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - hrint(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hrint(const __half h);

/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Truncate \p half2 vector input argument to the integral part.
* 
* \details Round each component of vector \p h to the largest integer value that does
* not exceed \p h in magnitude.
* \param[in] h - half2. Is only being read. 
* 
* \returns half2
* - The truncated \p h. 
*
* \see htrunc(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2trunc(const __half2 h);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculate \p half2 vector ceiling of the input argument.
* 
* \details For each component of vector \p h compute the smallest integer value not less
* than \p h.
* \param[in] h - half2. Is only being read. 
* 
* \returns half2
* - The vector of smallest integers not less than \p h. 
*
* \see hceil(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2ceil(const __half2 h);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculate the largest integer less than or equal to \p h.
* 
* \details For each component of vector \p h calculate the largest integer value which
* is less than or equal to \p h.
* \param[in] h - half2. Is only being read. 
* 
* \returns half2
* - The vector of largest integers which is less than or equal to \p h. 
*
* \see hfloor(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2floor(const __half2 h);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Round input to nearest integer value in half-precision floating-point
* number.
* 
* \details Round each component of \p half2 vector \p h to the nearest integer value in
* half-precision floating-point format, with halfway cases rounded to the
* nearest even integer value.
* \param[in] h - half2. Is only being read. 
* 
* \returns half2
* - The vector of rounded integer values. 
*
* \see hrint(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2rint(const __half2 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Returns \p half2 with both halves equal to the input value.
* 
* \details Returns \p half2 number with both halves equal to the input \p a \p half
* number.
* \param[in] a - half. Is only being read. 
* 
* \returns half2
* - The vector which has both its halves equal to the input \p a. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __half2half2(const __half a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Swaps both halves of the \p half2 input.
* 
* \details Swaps both halves of the \p half2 input and returns a new \p half2 number
* with swapped halves.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* - \p a with its halves being swapped. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __lowhigh2highlow(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts low 16 bits from each of the two \p half2 inputs and combines
* into one \p half2 number. 
* 
* \details Extracts low 16 bits from each of the two \p half2 inputs and combines into
* one \p half2 number. Low 16 bits from input \p a is stored in low 16 bits of
* the return value, low 16 bits from input \p b is stored in high 16 bits of
* the return value. 
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* - The low 16 bits of \p a and of \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __lows2half2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts high 16 bits from each of the two \p half2 inputs and
* combines into one \p half2 number.
* 
* \details Extracts high 16 bits from each of the two \p half2 inputs and combines into
* one \p half2 number. High 16 bits from input \p a is stored in low 16 bits of
* the return value, high 16 bits from input \p b is stored in high 16 bits of
* the return value.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* - The high 16 bits of \p a and of \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __highs2half2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Returns high 16 bits of \p half2 input.
*
* \details Returns high 16 bits of \p half2 input \p a.
* \param[in] a - half2. Is only being read. 
*
* \returns half
* - The high 16 bits of the input. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __high2half(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Returns low 16 bits of \p half2 input.
*
* \details Returns low 16 bits of \p half2 input \p a.
* \param[in] a - half2. Is only being read. 
*
* \returns half
* - Returns \p half which contains low 16 bits of the input \p a. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __low2half(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Checks if the input \p half number is infinite.
* 
* \details Checks if the input \p half number \p a is infinite. 
* \param[in] a - half. Is only being read. 
* 
* \returns int 
* - -1 if \p a is equal to negative infinity, 
* - 1 if \p a is equal to positive infinity, 
* - 0 otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ int __hisinf(const __half a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Combines two \p half numbers into one \p half2 number.
* 
* \details Combines two input \p half number \p a and \p b into one \p half2 number.
* Input \p a is stored in low 16 bits of the return value, input \p b is stored
* in high 16 bits of the return value.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
* 
* \returns half2
* - The half2 with one half equal to \p a and the other to \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __halves2half2(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts low 16 bits from \p half2 input.
* 
* \details Extracts low 16 bits from \p half2 input \p a and returns a new \p half2
* number which has both halves equal to the extracted bits.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* - The half2 with both halves equal to the low 16 bits of the input. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __low2half2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts high 16 bits from \p half2 input.
* 
* \details Extracts high 16 bits from \p half2 input \p a and returns a new \p half2
* number which has both halves equal to the extracted bits.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* - The half2 with both halves equal to the high 16 bits of the input. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __high2half2(const __half2 a);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in a \p half as a signed short integer.
* 
* \details Reinterprets the bits in the half-precision floating-point number \p h
* as a signed short integer. 
* \param[in] h - half. Is only being read. 
* 
* \returns short int
* - The reinterpreted value. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ short int __half_as_short(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in a \p half as an unsigned short integer.
* 
* \details Reinterprets the bits in the half-precision floating-point \p h
* as an unsigned short number.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned short int
* - The reinterpreted value.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned short int __half_as_ushort(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in a signed short integer as a \p half.
* 
* \details Reinterprets the bits in the signed short integer \p i as a
* half-precision floating-point number.
* \param[in] i - short int. Is only being read. 
* 
* \returns half
* - The reinterpreted value.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __short_as_half(const short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in an unsigned short integer as a \p half.
* 
* \details Reinterprets the bits in the unsigned short integer \p i as a
* half-precision floating-point number.
* \param[in] i - unsigned short int. Is only being read. 
* 
* \returns half
* - The reinterpreted value.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort_as_half(const unsigned short int i);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Calculates \p half maximum of two input values.
*
* \details Calculates \p half max(\p a, \p b)
* defined as (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmax(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Calculates \p half minimum of two input values.
*
* \details Calculates \p half min(\p a, \p b)
* defined as (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmin(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Calculates \p half2 vector maximum of two inputs.
*
* \details Calculates \p half2 vector max(\p a, \p b).
* Elementwise \p half operation is defined as
* (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* - The result of elementwise maximum of vectors \p a  and \p b
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmax2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Calculates \p half2 vector minimum of two inputs.
*
* \details Calculates \p half2 vector min(\p a, \p b).
* Elementwise \p half operation is defined as
* (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* - The result of elementwise minimum of vectors \p a  and \p b
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmin2(const __half2 a, const __half2 b);

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300)
#if !defined warpSize && !defined __local_warpSize
#define warpSize    32
#define __local_warpSize
#endif

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 700)

#if defined(_WIN32)
# define __CUDA_FP16_DEPRECATED__(msg) __declspec(deprecated(msg))
#elif (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__))))
# define __CUDA_FP16_DEPRECATED__(msg) __attribute__((deprecated))
#else
# define __CUDA_FP16_DEPRECATED__(msg) __attribute__((deprecated(msg)))
#endif

#if defined(_NVHPC_CUDA)
#define __CUDA_FP16_WSB_DEPRECATION_MESSAGE(x) __CUDA_FP16_STRINGIFY(x) "() is deprecated in favor of " __CUDA_FP16_STRINGIFY(x) "_sync() and may be removed in a future release."
#else
#define __CUDA_FP16_WSB_DEPRECATION_MESSAGE(x) __CUDA_FP16_STRINGIFY(x) "() is deprecated in favor of " __CUDA_FP16_STRINGIFY(x) "_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."
#endif

__CUDA_FP16_DECL__ __CUDA_FP16_DEPRECATED__(__CUDA_FP16_WSB_DEPRECATION_MESSAGE(__shfl)) __half2 __shfl(const __half2 var, const int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __CUDA_FP16_DEPRECATED__(__CUDA_FP16_WSB_DEPRECATION_MESSAGE(__shfl_up)) __half2 __shfl_up(const __half2 var, const unsigned int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __CUDA_FP16_DEPRECATED__(__CUDA_FP16_WSB_DEPRECATION_MESSAGE(__shfl_down))__half2 __shfl_down(const __half2 var, const unsigned int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __CUDA_FP16_DEPRECATED__(__CUDA_FP16_WSB_DEPRECATION_MESSAGE(__shfl_xor)) __half2 __shfl_xor(const __half2 var, const int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __CUDA_FP16_DEPRECATED__(__CUDA_FP16_WSB_DEPRECATION_MESSAGE(__shfl)) __half __shfl(const __half var, const int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __CUDA_FP16_DEPRECATED__(__CUDA_FP16_WSB_DEPRECATION_MESSAGE(__shfl_up)) __half __shfl_up(const __half var, const unsigned int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __CUDA_FP16_DEPRECATED__(__CUDA_FP16_WSB_DEPRECATION_MESSAGE(__shfl_down)) __half __shfl_down(const __half var, const unsigned int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __CUDA_FP16_DEPRECATED__(__CUDA_FP16_WSB_DEPRECATION_MESSAGE(__shfl_xor)) __half __shfl_xor(const __half var, const int delta, const int width = warpSize);

#undef __CUDA_FP16_WSB_DEPRECATION_MESSAGE
#undef __CUDA_FP16_DEPRECATED__
#endif /* !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700 */

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Direct copy from indexed thread. 
* 
* \details Returns the value of \p var held by the thread whose ID is given by \p srcLane. 
* If the \p width is less than \p warpSize, then each subsection of the warp behaves as a separate 
* entity with a starting logical thread ID of 0. If \p srcLane is outside the range \p [0:width-1], 
* the value returned corresponds to the value of \p var held by the \p srcLane modulo \p width (i.e. 
* within the same subsection). \p width must have a value which is a power of 2; 
* results are undefined if \p width is not a power of 2, or is a number greater than 
* \p warpSize.
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read. 
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - half2. Is only being read. 
* \param[in] srcLane - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 4-byte word referenced by \p var from the source thread ID as \p half2. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __shfl_sync(const unsigned int mask, const __half2 var, const int srcLane, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with lower ID relative to the caller. 
* 
* \details Calculates a source thread ID by subtracting \p delta from the caller's lane ID. 
* The value of \p var held by the resulting lane ID is returned: in effect, \p var is shifted up 
* the warp by \p delta threads. If the \p width is less than \p warpSize, then each subsection of the warp 
* behaves as a separate entity with a starting logical thread ID of 0. The source thread index 
* will not wrap around the value of \p width, so effectively the lower \p delta threads will be unchanged. 
* \p width must have a value which is a power of 2; results are undefined if \p width is not a power of 2, 
* or is a number greater than \p warpSize. 
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read. 
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - half2. Is only being read. 
* \param[in] delta - unsigned int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 4-byte word referenced by \p var from the source thread ID as \p half2. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __shfl_up_sync(const unsigned int mask, const __half2 var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with higher ID relative to the caller. 
* 
* \details Calculates a source thread ID by adding \p delta to the caller's thread ID. 
* The value of \p var held by the resulting thread ID is returned: this has the effect 
* of shifting \p var down the warp by \p delta threads. If the \p width is less than \p warpSize, then 
* each subsection of the warp behaves as a separate entity with a starting logical 
* thread ID of 0. Similarly to the __shfl_up_sync(), the ID number of the source thread 
* will not wrap around the value of \p width and the upper \p delta threads 
* will remain unchanged. 
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - half2. Is only being read. 
* \param[in] delta - unsigned int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 4-byte word referenced by \p var from the source thread ID as \p half2. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __shfl_down_sync(const unsigned int mask, const __half2 var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread based on bitwise XOR of own thread ID. 
* 
* \details Calculates a source thread ID by performing a bitwise XOR of the caller's thread ID with \p laneMask: 
* the value of \p var held by the resulting thread ID is returned. If the \p width is less than \p warpSize, then each 
* group of \p width consecutive threads are able to access elements from earlier groups of threads, 
* however if they attempt to access elements from later groups of threads their own value of \p var 
* will be returned. This mode implements a butterfly addressing pattern such as is used in tree 
* reduction and broadcast. 
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - half2. Is only being read. 
* \param[in] laneMask - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 4-byte word referenced by \p var from the source thread ID as \p half2. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __shfl_xor_sync(const unsigned int mask, const __half2 var, const int laneMask, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Direct copy from indexed thread. 
* 
* \details Returns the value of \p var held by the thread whose ID is given by \p srcLane. 
* If the \p width is less than \p warpSize, then each subsection of the warp behaves as a separate 
* entity with a starting logical thread ID of 0. If \p srcLane is outside the range \p [0:width-1], 
* the value returned corresponds to the value of \p var held by the \p srcLane modulo \p width (i.e. 
* within the same subsection). \p width must have a value which is a power of 2; 
* results are undefined if \p width is not a power of 2, or is a number greater than 
* \p warpSize.
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - half. Is only being read. 
* \param[in] srcLane - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 2-byte word referenced by \p var from the source thread ID as \p half. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __shfl_sync(const unsigned int mask, const __half var, const int srcLane, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with lower ID relative to the caller. 
* 
* \details Calculates a source thread ID by subtracting \p delta from the caller's lane ID. 
* The value of \p var held by the resulting lane ID is returned: in effect, \p var is shifted up 
* the warp by \p delta threads. If the \p width is less than \p warpSize, then each subsection of the warp 
* behaves as a separate entity with a starting logical thread ID of 0. The source thread index 
* will not wrap around the value of \p width, so effectively the lower \p delta threads will be unchanged. 
* \p width must have a value which is a power of 2; results are undefined if \p width is not a power of 2, 
* or is a number greater than \p warpSize. 
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - half. Is only being read. 
* \param[in] delta - unsigned int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 2-byte word referenced by \p var from the source thread ID as \p half. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __shfl_up_sync(const unsigned int mask, const __half var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with higher ID relative to the caller. 
* 
* \details Calculates a source thread ID by adding \p delta to the caller's thread ID. 
* The value of \p var held by the resulting thread ID is returned: this has the effect 
* of shifting \p var down the warp by \p delta threads. If the \p width is less than \p warpSize, then 
* each subsection of the warp behaves as a separate entity with a starting logical 
* thread ID of 0. Similarly to the __shfl_up_sync(), the ID number of the source thread 
* will not wrap around the value of \p width and the upper \p delta threads 
* will remain unchanged. 
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - half. Is only being read. 
* \param[in] delta - unsigned int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 2-byte word referenced by \p var from the source thread ID as \p half. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __shfl_down_sync(const unsigned int mask, const __half var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread based on bitwise XOR of own thread ID. 
* 
* \details Calculates a source thread ID by performing a bitwise XOR of the caller's thread ID with \p laneMask: 
* the value of \p var held by the resulting thread ID is returned. If the \p width is less than \p warpSize, then each 
* group of \p width consecutive threads are able to access elements from earlier groups of threads, 
* however if they attempt to access elements from later groups of threads their own value of \p var 
* will be returned. This mode implements a butterfly addressing pattern such as is used in tree 
* reduction and broadcast. 
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - half. Is only being read. 
* \param[in] laneMask - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 2-byte word referenced by \p var from the source thread ID as \p half. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __shfl_xor_sync(const unsigned int mask, const __half var, const int laneMask, const int width = warpSize);

#if defined(__local_warpSize)
#undef warpSize
#undef __local_warpSize
#endif
#endif /*!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300) */

#if defined(__cplusplus) && ( !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 320) )
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.nc` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldg(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.nc` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldg(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cg` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldcg(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cg` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldcg(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.ca` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldca(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.ca` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldca(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cs` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldcs(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cs` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldcs(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.lu` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldlu(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.lu` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldlu(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cv` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldcv(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cv` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldcv(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.wb` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stwb(__half2 *const ptr, const __half2 value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.wb` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stwb(__half *const ptr, const __half value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.cg` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stcg(__half2 *const ptr, const __half2 value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.cg` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stcg(__half *const ptr, const __half value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.cs` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stcs(__half2 *const ptr, const __half2 value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.cs` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stcs(__half *const ptr, const __half value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.wt` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stwt(__half2 *const ptr, const __half2 value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.wt` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stwt(__half *const ptr, const __half value);
#endif /*defined(__cplusplus) && ( !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 320) )*/
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs half2 vector if-equal comparison.
* 
* \details Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* - The vector result of if-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __heq2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector not-equal comparison.
* 
* \details Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* - The vector result of not-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hne2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-equal comparison.
*
* \details Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The \p half2 result of less-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hle2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-equal comparison.
*
* \details Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The vector result of greater-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hge2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-than comparison.
*
* \details Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The half2 vector result of less-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hlt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-than comparison.
* 
* \details Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* - The vector result of greater-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hgt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered if-equal comparison.
* 
* \details Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* - The vector result of unordered if-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hequ2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered not-equal comparison.
*
* \details Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The vector result of unordered not-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hneu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-equal comparison.
*
* Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The vector result of unordered less-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hleu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-equal comparison.
*
* \details Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The \p half2 vector result of unordered greater-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hgeu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-than comparison.
*
* \details Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The vector result of unordered less-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hltu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-than comparison.
*
* \details Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The \p half2 vector result of unordered greater-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hgtu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs half2 vector if-equal comparison.
* 
* \details Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns unsigned int
* - The vector mask result of if-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __heq2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector not-equal comparison.
* 
* \details Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns unsigned int
* - The vector mask result of not-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hne2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-equal comparison.
*
* \details Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns unsigned int
* - The vector mask result of less-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hle2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-equal comparison.
*
* \details Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns unsigned int
* - The vector mask result of greater-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hge2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-than comparison.
*
* \details Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns unsigned int
* - The vector mask result of less-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hlt2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-than comparison.
* 
* \details Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns unsigned int
* - The vector mask result of greater-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hgt2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered if-equal comparison.
* 
* \details Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns unsigned int
* - The vector mask result of unordered if-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hequ2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered not-equal comparison.
*
* \details Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns unsigned int
* - The vector mask result of unordered not-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hneu2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-equal comparison.
*
* Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns unsigned int
* - The vector mask result of unordered less-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hleu2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-equal comparison.
*
* \details Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns unsigned int
* - The vector mask result of unordered greater-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hgeu2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-than comparison.
*
* \details Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns unsigned int
* - The vector mask result of unordered less-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hltu2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-than comparison.
*
* \details Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to \p 0xFFFF for true, or \p 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns unsigned int
* - The vector mask result of unordered greater-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hgtu2_mask(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Determine whether \p half2 argument is a NaN.
*
* \details Determine whether each half of input \p half2 number \p a is a NaN.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - The half2 with the corresponding \p half results set to
* 1.0 for NaN, 0.0 otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hisnan2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector addition in round-to-nearest-even mode.
*
* \details Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-95
* \endinternal
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The sum of vectors \p a and \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hadd2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p half2 input vector \p b from input vector \p a in
* round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-104
* \endinternal
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The subtraction of vector \p b from \p a. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hsub2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector multiplication in round-to-nearest-even mode.
*
* \details Performs \p half2 vector multiplication of inputs \p a and \p b, in
* round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-102
* \endinternal
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The result of elementwise multiplying the vectors \p a and \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmul2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector addition in round-to-nearest-even mode.
*
* \details Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest-even
* mode. Prevents floating-point contractions of mul+add into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-95
* \endinternal
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* - The sum of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hadd2_rn(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p half2 input vector \p b from input vector \p a in
* round-to-nearest-even mode. Prevents floating-point contractions of mul+sub
* into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-104
* \endinternal
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* - The subtraction of vector \p b from \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hsub2_rn(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector multiplication in round-to-nearest-even mode.
*
* \details Performs \p half2 vector multiplication of inputs \p a and \p b, in
* round-to-nearest-even mode. Prevents floating-point contractions of
* mul+add or sub into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-102
* \endinternal
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* - The result of elementwise multiplying the vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmul2_rn(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector division in round-to-nearest-even mode.
*
* \details Divides \p half2 input vector \p a by input vector \p b in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-103
* \endinternal
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The elementwise division of \p a with \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __h2div(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Calculates the absolute value of both halves of the input \p half2 number and
* returns the result.
*
* \details Calculates the absolute value of both halves of the input \p half2 number and
* returns the result.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - Returns \p a with the absolute value of both halves. 
*
* \see __habs(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __habs2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector addition in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest-even
* mode, and clamps the results to range [0.0, 1.0]. NaN results are flushed to
* +0.0.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The sum of \p a and \p b, with respect to saturation. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hadd2_sat(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector subtraction in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* \details Subtracts \p half2 input vector \p b from input vector \p a in
* round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
* results are flushed to +0.0.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The subtraction of vector \p b from \p a, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hsub2_sat(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector multiplication in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* \details Performs \p half2 vector multiplication of inputs \p a and \p b, in
* round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
* results are flushed to +0.0.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* - The result of elementwise multiplication of vectors \p a and \p b, 
* with respect to saturation. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmul2_sat(const __half2 a, const __half2 b);

#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
* mode.
*
* \details Performs \p half2 vector multiply on inputs \p a and \p b,
* then performs a \p half2 vector add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-105
* \endinternal
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* \param[in] c - half2. Is only being read. 
*
* \returns half2
* - The result of elementwise fused multiply-add operation on vectors \p a, \p b, and \p c. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
* mode, with saturation to [0.0, 1.0].
*
* \details Performs \p half2 vector multiply on inputs \p a and \p b,
* then performs a \p half2 vector add of the result with \p c,
* rounding the result once in round-to-nearest-even mode, and clamps the
* results to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* \param[in] c - half2. Is only being read. 
*
* \returns half2
* - The result of elementwise fused multiply-add operation on vectors \p a, \p b, and \p c, 
* with respect to saturation. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c);
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Negates both halves of the input \p half2 number and returns the
* result.
*
* \details Negates both halves of the input \p half2 number \p a and returns the result.
* \internal
* \req DEEPLEARN-SRM_REQ-101
* \endinternal
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - Returns \p a with both halves negated. 
* 
* \see __hneg(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hneg2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Calculates the absolute value of input \p half number and returns the result.
*
* \details Calculates the absolute value of input \p half number and returns the result.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The absolute value of \p a.
* - __habs \cuda_math_formula (\pm 0)\end_cuda_math_formula returns +0.
* - __habs \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula +\infty \end_cuda_math_formula.
* - __habs(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __habs(const __half a);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half addition in round-to-nearest-even mode.
*
* \details Performs \p half addition of inputs \p a and \p b, in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-94
* \endinternal
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* - The sum of \p a and \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hadd(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p half input \p b from input \p a in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-97
* \endinternal
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* - The result of subtracting \p b from \p a. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hsub(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half multiplication in round-to-nearest-even mode.
*
* \details Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-99
* \endinternal
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* - The result of multiplying \p a and \p b. 
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmul(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half addition in round-to-nearest-even mode.
*
* \details Performs \p half addition of inputs \p a and \p b, in round-to-nearest-even
* mode. Prevents floating-point contractions of mul+add into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-94
* \endinternal
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* - The sum of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hadd_rn(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p half input \p b from input \p a in round-to-nearest-even
* mode. Prevents floating-point contractions of mul+sub into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-97
* \endinternal
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* - The result of subtracting \p b from \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hsub_rn(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half multiplication in round-to-nearest-even mode.
*
* \details Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest-even
* mode. Prevents floating-point contractions of mul+add or sub into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-99
* \endinternal
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* - The result of multiplying \p a and \p b.
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmul_rn(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half division in round-to-nearest-even mode.
* 
* \details Divides \p half input \p a by input \p b in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-98
* \endinternal
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
* 
* \returns half
* - The result of dividing \p a by \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__  __half __hdiv(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half addition in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Performs \p half add of inputs \p a and \p b, in round-to-nearest-even mode,
* and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* - The sum of \p a and \p b, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hadd_sat(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half subtraction in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Subtracts \p half input \p b from input \p a in round-to-nearest-even
* mode,
* and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* - The result of subtraction of \p b from \p a, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hsub_sat(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half multiplication in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest-even
* mode, and clamps the result to range [0.0, 1.0]. NaN results are flushed to
* +0.0.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* - The result of multiplying \p a and \p b, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmul_sat(const __half a, const __half b);

#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half fused multiply-add in round-to-nearest-even mode.
*
* \details Performs \p half multiply on inputs \p a and \p b,
* then performs a \p half add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-96
* \endinternal
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
* \param[in] c - half. Is only being read. 
*
* \returns half
* - The result of fused multiply-add operation on \p
* a, \p b, and \p c. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half fused multiply-add in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* \details Performs \p half multiply on inputs \p a and \p b,
* then performs a \p half add of the result with \p c,
* rounding the result once in round-to-nearest-even mode, and clamps the result
* to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
* \param[in] c - half. Is only being read. 
*
* \returns half
* - The result of fused multiply-add operation on \p
* a, \p b, and \p c, with respect to saturation. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hfma_sat(const __half a, const __half b, const __half c);
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */

/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Negates input \p half number and returns the result.
*
* \details Negates input \p half number and returns the result.
* \internal
* \req DEEPLEARN-SRM_REQ-100
* \endinternal
* \param[in] a - half. Is only being read. 
*
* \returns half
* - Negated input \p a.
* - __hneg \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \mp 0 \end_cuda_math_formula.
* - __hneg \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \mp \infty \end_cuda_math_formula.
* - __hneg(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hneg(const __half a);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector if-equal comparison and returns boolean true
* if both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half if-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* - true if both \p half results of if-equal comparison
* of vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbeq2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector not-equal comparison and returns boolean
* true if both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half not-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* - true if both \p half results of not-equal comparison
* of vectors \p a and \p b are true, 
* - false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbne2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-equal comparison and returns boolean
* true if both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* - true if both \p half results of less-equal comparison
* of vectors \p a and \p b are true; 
* - false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hble2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-equal comparison and returns boolean
* true if both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* - true if both \p half results of greater-equal
* comparison of vectors \p a and \p b are true; 
* - false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbge2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-than comparison and returns boolean
* true if both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* - true if both \p half results of less-than comparison
* of vectors \p a and \p b are true; 
* - false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hblt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-than comparison and returns boolean
* true if both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns bool 
* - true if both \p half results of greater-than
* comparison of vectors \p a and \p b are true; 
* - false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbgt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered if-equal comparison and returns
* boolean true if both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half if-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* - true if both \p half results of unordered if-equal
* comparison of vectors \p a and \p b are true; 
* - false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbequ2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered not-equal comparison and returns
* boolean true if both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half not-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* - true if both \p half results of unordered not-equal
* comparison of vectors \p a and \p b are true;
* - false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbneu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-equal comparison and returns
* boolean true if both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* - true if both \p half results of unordered less-equal
* comparison of vectors \p a and \p b are true; 
* - false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbleu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-equal comparison and
* returns boolean true if both \p half results are true, boolean false
* otherwise.
*
* \details Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* - true if both \p half results of unordered
* greater-equal comparison of vectors \p a and \p b are true; 
* - false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbgeu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-than comparison and returns
* boolean true if both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* - true if both \p half results of unordered less-than comparison of 
* vectors \p a and \p b are true; 
* - false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbltu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-than comparison and
* returns boolean true if both \p half results are true, boolean false
* otherwise.
*
* \details Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* - true if both \p half results of unordered
* greater-than comparison of vectors \p a and \p b are true;
* - false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbgtu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half if-equal comparison.
*
* \details Performs \p half if-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of if-equal comparison of \p a and \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __heq(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half not-equal comparison.
*
* \details Performs \p half not-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of not-equal comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hne(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half less-equal comparison.
*
* \details Performs \p half less-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of less-equal comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hle(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half greater-equal comparison.
*
* \details Performs \p half greater-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of greater-equal comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hge(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half less-than comparison.
*
* \details Performs \p half less-than comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of less-than comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hlt(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half greater-than comparison.
*
* \details Performs \p half greater-than comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of greater-than comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hgt(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered if-equal comparison.
*
* \details Performs \p half if-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of unordered if-equal comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hequ(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered not-equal comparison.
*
* \details Performs \p half not-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of unordered not-equal comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hneu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered less-equal comparison.
*
* \details Performs \p half less-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of unordered less-equal comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hleu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered greater-equal comparison.
*
* \details Performs \p half greater-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of unordered greater-equal comparison of \p a
* and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hgeu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered less-than comparison.
*
* \details Performs \p half less-than comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of unordered less-than comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hltu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered greater-than comparison.
*
* \details Performs \p half greater-than comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* - The boolean result of unordered greater-than comparison of \p a
* and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hgtu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Determine whether \p half argument is a NaN.
*
* \details Determine whether \p half value \p a is a NaN.
* \param[in] a - half. Is only being read. 
*
* \returns bool
* - true if argument is NaN. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hisnan(const __half a);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Calculates \p half maximum of two input values, NaNs pass through.
*
* \details Calculates \p half max(\p a, \p b)
* defined as (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmax_nan(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Calculates \p half minimum of two input values, NaNs pass through.
*
* \details Calculates \p half min(\p a, \p b)
* defined as (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmin_nan(const __half a, const __half b);
#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half fused multiply-add in round-to-nearest-even mode with relu saturation.
*
* \details Performs \p half multiply on inputs \p a and \p b,
* then performs a \p half add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* Then negative result is clamped to 0.
* NaN result is converted to canonical NaN.
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
* \param[in] c - half. Is only being read.
*
* \returns half
* - The result of fused multiply-add operation on \p
* a, \p b, and \p c with relu saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hfma_relu(const __half a, const __half b, const __half c);
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Calculates \p half2 vector maximum of two inputs, NaNs pass through.
*
* \details Calculates \p half2 vector max(\p a, \p b).
* Elementwise \p half operation is defined as
* (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* - The result of elementwise maximum of vectors \p a  and \p b, with NaNs pass through
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmax2_nan(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Calculates \p half2 vector minimum of two inputs, NaNs pass through.
*
* \details Calculates \p half2 vector min(\p a, \p b).
* Elementwise \p half operation is defined as
* (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* - The result of elementwise minimum of vectors \p a  and \p b, with NaNs pass through
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmin2_nan(const __half2 a, const __half2 b);
#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
* mode with relu saturation.
*
* \details Performs \p half2 vector multiply on inputs \p a and \p b,
* then performs a \p half2 vector add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* Then negative result is clamped to 0.
* NaN result is converted to canonical NaN.
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
* \param[in] c - half2. Is only being read.
*
* \returns half2
* - The result of elementwise fused multiply-add operation on vectors \p a, \p b, and \p c with relu saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hfma2_relu(const __half2 a, const __half2 b, const __half2 c);

/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs fast complex multiply-accumulate
*
* \details Interprets vector \p half2 input pairs \p a, \p b, and \p c as
* complex numbers in \p half precision: (a.x + I*a.y), (b.x + I*b.y), (c.x + I*c.y)
* and performs complex multiply-accumulate operation: a*b + c in a simple way:
* ((a.x*b.x + c.x) - a.y*b.y) + I*((a.x*b.y + c.y) + a.y*b.x)
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
* \param[in] c - half2. Is only being read.
*
* \returns half2
* - The result of complex multiply-accumulate operation on complex numbers \p a, \p b, and \p c
* - __half2 result = __hcmadd(a, b, c) is numerically in agreement with:
* - result.x = __hfma(-a.y, b.y, __hfma(a.x, b.x, c.x))
* - result.y = __hfma( a.y, b.x, __hfma(a.x, b.y, c.y))
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hcmadd(const __half2 a, const __half2 b, const __half2 c);
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half square root in round-to-nearest-even mode.
*
* \details Calculates \p half square root of input: \cuda_math_formula \sqrt{a} \end_cuda_math_formula in round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The square root of \p a.
* - hsqrt \cuda_math_formula (+\infty)\end_cuda_math_formula returns \cuda_math_formula +\infty \end_cuda_math_formula.
* - hsqrt \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - hsqrt \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns NaN.
* - hsqrt(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hsqrt(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half reciprocal square root in round-to-nearest-even
* mode.
*
* \details Calculates \p half reciprocal square root of input: \cuda_math_formula \frac{1}{\sqrt{a}}\end_cuda_math_formula in round-to-nearest-even
* mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The reciprocal square root of \p a.
* - hrsqrt \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - hrsqrt \cuda_math_formula (+\infty)\end_cuda_math_formula returns +0.
* - hrsqrt \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns NaN.
* - hrsqrt(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hrsqrt(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half reciprocal in round-to-nearest-even mode.
*
* \details Calculates \p half reciprocal of input: \cuda_math_formula \frac{1}{a}\end_cuda_math_formula in round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The reciprocal of \p a.
* - hrcp \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - hrcp \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - hrcp(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hrcp(const __half a);
#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half natural logarithm in round-to-nearest-even mode.
*
* \details Calculates \p half natural logarithm of input: \cuda_math_formula \ln(a)\end_cuda_math_formula in round-to-nearest-even
* mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The natural logarithm of \p a.
* - hlog \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula -\infty \end_cuda_math_formula.
* - hlog(1) returns +0.
* - hlog(x), x < 0 returns NaN.
* - hlog \cuda_math_formula (+\infty)\end_cuda_math_formula returns \cuda_math_formula +\infty \end_cuda_math_formula.
* - hlog(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hlog(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half binary logarithm in round-to-nearest-even mode.
*
* \details Calculates \p half binary logarithm of input: \cuda_math_formula \log_{2}(a)\end_cuda_math_formula in round-to-nearest-even
* mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The binary logarithm of \p a.
* - hlog2 \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula -\infty \end_cuda_math_formula.
* - hlog2(1) returns +0.
* - hlog2(x), x < 0 returns NaN.
* - hlog2 \cuda_math_formula (+\infty)\end_cuda_math_formula returns \cuda_math_formula +\infty \end_cuda_math_formula.
* - hlog2(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hlog2(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half decimal logarithm in round-to-nearest-even mode.
*
* \details Calculates \p half decimal logarithm of input: \cuda_math_formula \log_{10}(a)\end_cuda_math_formula in round-to-nearest-even
* mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The decimal logarithm of \p a.
* - hlog10 \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula -\infty \end_cuda_math_formula.
* - hlog10(1) returns +0.
* - hlog10(x), x < 0 returns NaN.
* - hlog10 \cuda_math_formula (+\infty)\end_cuda_math_formula returns \cuda_math_formula +\infty \end_cuda_math_formula.
* - hlog10(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hlog10(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half natural exponential function in round-to-nearest-even
* mode.
*
* \details Calculates \p half natural exponential function of input: \cuda_math_formula e^{a}\end_cuda_math_formula in
* round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The natural exponential function on \p a.
* - hexp \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 1.
* - hexp \cuda_math_formula (-\infty)\end_cuda_math_formula returns +0.
* - hexp \cuda_math_formula (+\infty)\end_cuda_math_formula returns \cuda_math_formula +\infty \end_cuda_math_formula.
* - hexp(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hexp(const __half a);
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */

/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates approximate \p half hyperbolic tangent function.
*
* \details Calculates approximate \p half hyperbolic tangent function: \cuda_math_formula \tanh(a)\end_cuda_math_formula.
* This operation uses HW acceleration on devices of compute capability 7.5 and higher.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The approximate hyperbolic tangent function of \p a.
* - htanh_approx \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula (\pm 0)\end_cuda_math_formula.
* - htanh_approx \cuda_math_formula (\pm\infty)\end_cuda_math_formula returns \cuda_math_formula (\pm 1)\end_cuda_math_formula.
* - htanh_approx(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half htanh_approx(const __half a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector approximate hyperbolic tangent function.
*
* \details Calculates \p half2 approximate hyperbolic tangent function of input vector \p a.
* This operation uses HW acceleration on devices of compute capability 7.5 and higher.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - The elementwise approximate hyperbolic tangent function on vector \p a.
* 
* \see htanh_approx(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2tanh_approx(const __half2 a);

/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half hyperbolic tangent function in
* round-to-nearest-even mode.
*
* \details Calculates \p half hyperbolic tangent function: \cuda_math_formula \tanh(a)\end_cuda_math_formula in
* round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The hyperbolic tangent function of \p a.
* - htanh \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula (\pm 0)\end_cuda_math_formula.
* - htanh \cuda_math_formula (\pm\infty)\end_cuda_math_formula returns \cuda_math_formula (\pm 1)\end_cuda_math_formula.
* - htanh(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half htanh(const __half a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector hyperbolic tangent function in round-to-nearest-even
* mode.
*
* \details Calculates \p half2 hyperbolic tangent function of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - The elementwise hyperbolic tangent function on vector \p a.
* 
* \see htanh(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2tanh(const __half2 a);

/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half binary exponential function in round-to-nearest-even
* mode.
*
* \details Calculates \p half binary exponential function of input: \cuda_math_formula 2^{a}\end_cuda_math_formula in
* round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The binary exponential function on \p a.
* - hexp2 \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 1.
* - hexp2 \cuda_math_formula (-\infty)\end_cuda_math_formula returns +0.
* - hexp2 \cuda_math_formula (+\infty)\end_cuda_math_formula returns \cuda_math_formula +\infty \end_cuda_math_formula.
* - hexp2(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hexp2(const __half a);
#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half decimal exponential function in round-to-nearest-even
* mode.
*
* \details Calculates \p half decimal exponential function of input: \cuda_math_formula 10^{a}\end_cuda_math_formula in
* round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The decimal exponential function on \p a.
* - hexp10 \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 1.
* - hexp10 \cuda_math_formula (-\infty)\end_cuda_math_formula returns +0.
* - hexp10 \cuda_math_formula (+\infty)\end_cuda_math_formula returns \cuda_math_formula +\infty \end_cuda_math_formula.
* - hexp10(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hexp10(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half cosine in round-to-nearest-even mode.
*
* \details Calculates \p half cosine of input \p a in round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The cosine of \p a.
* - hcos \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 1.
* - hcos \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns NaN.
* - hcos(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hcos(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half sine in round-to-nearest-even mode.
*
* \details Calculates \p half sine of input \p a in round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* - The sine of \p a.
* - hsin \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula (\pm 0)\end_cuda_math_formula.
* - hsin \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns NaN.
* - hsin(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hsin(const __half a);
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector square root in round-to-nearest-even mode.
*
* \details Calculates \p half2 square root of input vector \p a in round-to-nearest-even
* mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - The elementwise square root on vector \p a.
* 
* \see hsqrt(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2sqrt(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector reciprocal square root in round-to-nearest-even
* mode.
*
* \details Calculates \p half2 reciprocal square root of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - The elementwise reciprocal square root on vector \p a.
* 
* \see hrsqrt(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2rsqrt(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector reciprocal in round-to-nearest-even mode.
*
* \details Calculates \p half2 reciprocal of input vector \p a in round-to-nearest-even
* mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - The elementwise reciprocal on vector \p a.
* 
* \see hrcp(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2rcp(const __half2 a);
#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector natural logarithm in round-to-nearest-even
* mode.
*
* \details Calculates \p half2 natural logarithm of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - The elementwise natural logarithm on vector \p a.
* 
* \see hlog(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2log(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector binary logarithm in round-to-nearest-even
* mode.
*
* \details Calculates \p half2 binary logarithm of input vector \p a in round-to-nearest-even
* mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - The elementwise binary logarithm on vector \p a.
* 
* \see hlog2(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2log2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector decimal logarithm in round-to-nearest-even
* mode.
*
* \details Calculates \p half2 decimal logarithm of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - The elementwise decimal logarithm on vector \p a.
* 
* \see hlog10(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2log10(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector exponential function in round-to-nearest-even
* mode.
*
* \details Calculates \p half2 exponential function of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - The elementwise exponential function on vector \p a.
* 
* \see hexp(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2exp(const __half2 a);
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector binary exponential function in
* round-to-nearest-even mode.
*
* \details Calculates \p half2 binary exponential function of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* - The elementwise binary exponential function on vector \p a.
* 
* \see hexp2(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2exp2(const __half2 a);
#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector decimal exponential function in
* round-to-nearest-even mode.
* 
* \details Calculates \p half2 decimal exponential function of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* - The elementwise decimal exponential function on vector \p a.
* 
* \see hexp10(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2exp10(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector cosine in round-to-nearest-even mode.
* 
* \details Calculates \p half2 cosine of input vector \p a in round-to-nearest-even
* mode.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* - The elementwise cosine on vector \p a.
* 
* \see hcos(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2cos(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector sine in round-to-nearest-even mode.
* 
* \details Calculates \p half2 sine of input vector \p a in round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* - The elementwise sine on vector \p a.
* 
* \see hsin(__half) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2sin(const __half2 a);
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */

/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Vector add \p val to the value stored at \p address in global or shared memory, and writes this
* value back to \p address. The atomicity of the add operation is guaranteed separately for each of the
* two \p __half elements; the entire \p __half2 is not guaranteed to be atomic as a single 32-bit access.
* 
* \details The location of \p address must be in global or shared memory. This operation has undefined
* behavior otherwise. This operation is natively supported by devices of compute capability 6.x and higher,
* older devices use emulation path.
* 
* \param[in] address - half2*. An address in global or shared memory.
* \param[in] val - half2. The value to be added.
* 
* \returns half2
* - The old value read from \p address.
*
* \note_ref_guide_atomic
*/
__CUDA_FP16_DECL__ __half2 atomicAdd(__half2 *const address, const __half2 val);

#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Adds \p val to the value stored at \p address in global or shared memory, and writes this value
* back to \p address. This operation is performed in one atomic operation.
* 
* \details The location of \p address must be in global or shared memory. This operation has undefined
* behavior otherwise. This operation is only supported by devices of compute capability 7.x and higher.
* 
* \param[in] address - half*. An address in global or shared memory.
* \param[in] val - half. The value to be added.
* 
* \returns half
* - The old value read from \p address.
* 
* \note_ref_guide_atomic
*/
__CUDA_FP16_DECL__ __half atomicAdd(__half *const address, const __half val);
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA) */
#endif /*defined(__CUDACC__) || defined(_NVHPC_CUDA)*/


#endif /* defined(__cplusplus) */

#if !defined(_MSC_VER) && __cplusplus >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_FP16
#elif _MSC_FULL_VER >= 190024210 && _MSVC_LANG >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_FP16
#endif

// implicitly provided by NVRTC
#if !defined(__CUDACC_RTC__)
#include <nv/target>
#endif  /* !defined(__CUDACC_RTC__) */

/* C++11 header for std::move. 
 * In RTC mode, std::move is provided implicitly; don't include the header
 */
#if defined(__CPP_VERSION_AT_LEAST_11_FP16) && !defined(__CUDACC_RTC__)
#include <utility>
#endif /* __cplusplus >= 201103L && !defined(__CUDACC_RTC__) */

/* C++ header for std::memcpy (used for type punning in host-side implementations).
 * When compiling as a CUDA source file memcpy is provided implicitly.
 * !defined(__CUDACC__) implies !defined(__CUDACC_RTC__).
 */
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <cstring>
#endif /* defined(__cplusplus) && !defined(__CUDACC__) */

#if (defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3))))
#define __CUDA_FP16_INLINE__
#define __CUDA_FP16_FORCEINLINE__
#else
#define __CUDA_FP16_INLINE__ inline
#define __CUDA_FP16_FORCEINLINE__ __forceinline__
#endif /* (defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3)))) */

/* Set up structure-alignment attribute */
#if defined(__CUDACC__)
#define __CUDA_ALIGN__(align) __align__(align)
#else
/* Define alignment macro based on compiler type (cannot assume C11 "_Alignas" is available) */
#if __cplusplus >= 201103L
#define __CUDA_ALIGN__(n) alignas(n)    /* C++11 kindly gives us a keyword for this */
#else /* !defined(__CPP_VERSION_AT_LEAST_11_FP16)*/
#if defined(__GNUC__)
#define __CUDA_ALIGN__(n) __attribute__ ((aligned(n)))
#elif defined(_MSC_VER)
#define __CUDA_ALIGN__(n) __declspec(align(n))
#else
#define __CUDA_ALIGN__(n)
#endif /* defined(__GNUC__) */
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */
#endif /* defined(__CUDACC__) */

// define __CUDA_FP16_CONSTEXPR__ in order to
// use constexpr where possible, with supporting C++ dialects
// undef after use
#if (defined __CPP_VERSION_AT_LEAST_11_FP16)
#define __CUDA_FP16_CONSTEXPR__   constexpr
#else
#define __CUDA_FP16_CONSTEXPR__
#endif

/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief __half_raw data type
 * \details Type allows static initialization of \p half until it becomes
 * a built-in type.
 * 
 * - Note: this initialization is as a bit-field representation of \p half,
 * and not a conversion from \p short to \p half.
 * Such representation will be deprecated in a future version of CUDA.
 * 
 * - Note: this is visible to non-nvcc compilers, including C-only compilations
 */
typedef struct __CUDA_ALIGN__(2) {
    /**
     * Storage field contains bits representation of the \p half floating-point number.
     */
    unsigned short x;
} __half_raw;

/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief __half2_raw data type
 * \details Type allows static initialization of \p half2 until it becomes
 * a built-in type.
 * 
 * - Note: this initialization is as a bit-field representation of \p half2,
 * and not a conversion from \p short2 to \p half2.
 * Such representation will be deprecated in a future version of CUDA.
 * 
 * - Note: this is visible to non-nvcc compilers, including C-only compilations
 */
typedef struct __CUDA_ALIGN__(4) {
    /**
     * Storage field contains bits of the lower \p half part.
     */
    unsigned short x;
    /**
     * Storage field contains bits of the upper \p half part.
     */
    unsigned short y;
} __half2_raw;

/* All other definitions in this file are only visible to C++ compilers */
#if defined(__cplusplus)

/* Hide GCC member initialization list warnings because of host/device in-function init requirement */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

/* class' : multiple assignment operators specified
   The class has multiple assignment operators of a single type. This warning is informational */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( push )
#pragma warning( disable:4522 )
#endif /* defined(_MSC_VER) && _MSC_VER >= 1500 */

// forward-declaration of bfloat type to be used in converting constructor
struct __nv_bfloat16;

/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief __half data type
 * \details This structure implements the datatype for storing 
 * half-precision floating-point numbers. The structure implements 
 * assignment, arithmetic and comparison operators, and type conversions. 
 * 16 bits are being used in total: 1 sign bit, 5 bits for the exponent, 
 * and the significand is being stored in 10 bits. 
 * The total precision is 11 bits. There are 15361 representable 
 * numbers within the interval [0.0, 1.0], endpoints included. 
 * On average we have log10(2**11) ~ 3.311 decimal digits. 
 * 
 * The objective here is to provide IEEE754-compliant implementation
 * of \p binary16 type and arithmetic with limitations due to
 * device HW not supporting floating-point exceptions.
 */
struct __CUDA_ALIGN__(2) __half {
protected:
    /**
     * Protected storage variable contains the bits of floating-point data.
     */
    unsigned short __x;

public:
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * \brief Constructor by default.
     * \details Emtpy default constructor, result is uninitialized.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP16)
    __half() = default;
#else
    __CUDA_HOSTDEVICE__ __half() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */

    /* Convert to/from __half_raw */
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Constructor from \p __half_raw.
     */
    __CUDA_HOSTDEVICE__ __CUDA_FP16_CONSTEXPR__ __half(const __half_raw &hr) : __x(hr.x) { }
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Assignment operator from \p __half_raw.
     */
    __CUDA_HOSTDEVICE__ __half &operator=(const __half_raw &hr);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Assignment operator from \p __half_raw to \p volatile \p __half.
     */
    __CUDA_HOSTDEVICE__ volatile __half &operator=(const __half_raw &hr) volatile;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Assignment operator from \p volatile \p __half_raw to \p volatile \p __half.
     */
    __CUDA_HOSTDEVICE__ volatile __half &operator=(const volatile __half_raw &hr) volatile;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Type cast to \p __half_raw operator.
     */
    __CUDA_HOSTDEVICE__ operator __half_raw() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Type cast to \p __half_raw operator with \p volatile input.
     */
    __CUDA_HOSTDEVICE__ operator __half_raw() const volatile;
#if !defined(__CUDA_NO_HALF_CONVERSIONS__)
#if defined(__CPP_VERSION_AT_LEAST_11_FP16)
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Construct \p __half from \p __nv_bfloat16 input using default round-to-nearest-even rounding mode.
     * Need to include the header file \p cuda_bf16.h
     */
    explicit __CUDA_HOSTDEVICE__ __half(const __nv_bfloat16 f); //forward declaration only, implemented in cuda_bf16.hpp
#endif /* #if defined(__CPP_VERSION_AT_LEAST_11_FP16) */
    /* Construct from float/double */
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Construct \p __half from \p float input using default round-to-nearest-even rounding mode.
     *
     * \see __float2half(float) for further details.
     */
    __CUDA_HOSTDEVICE__ __half(const float f) { __x = __float2half(f).__x; }
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Construct \p __half from \p double input using default round-to-nearest-even rounding mode.
     *
     * \see __double2half(double) for further details.
     */
    __CUDA_HOSTDEVICE__ __half(const double f) { __x = __double2half(f).__x; }
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Type cast to \p float operator.
     */
    __CUDA_HOSTDEVICE__ operator float() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Type cast to \p __half assignment operator from \p float input using default round-to-nearest-even rounding mode.
     *
     * \see __float2half(float) for further details.
     */
    __CUDA_HOSTDEVICE__ __half &operator=(const float f);

    /* We omit "cast to double" operator, so as to not be ambiguous about up-cast */
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Type cast to \p __half assignment operator from \p double input using default round-to-nearest-even rounding mode.
     *
     * \see __double2half(double) for further details.
     */
    __CUDA_HOSTDEVICE__ __half &operator=(const double f);

/*
 * Implicit type conversions to/from integer types were only available to nvcc compilation.
 * Introducing them for all compilers is a potentially breaking change that may affect
 * overloads resolution and will require users to update their code.
 * Define __CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__ to opt-out.
 */
#if !(defined __CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__) || (defined __CUDACC__)
    /* Allow automatic construction from types supported natively in hardware */
    /* Note we do avoid constructor init-list because of special host/device compilation rules */

    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Construct \p __half from \p short integer input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half(const short val) { __x = __short2half_rn(val).__x; }
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Construct \p __half from \p unsigned \p short integer input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half(const unsigned short val) { __x = __ushort2half_rn(val).__x; }
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Construct \p __half from \p int input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half(const int val) { __x = __int2half_rn(val).__x; }
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Construct \p __half from \p unsigned \p int input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half(const unsigned int val) { __x = __uint2half_rn(val).__x; }
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Construct \p __half from \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half(const long val) {
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(long) == sizeof(long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (default: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            __x = __ll2half_rn(static_cast<long long>(val)).__x;
        } else {
            __x = __int2half_rn(static_cast<int>(val)).__x;
        }
    }
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Construct \p __half from \p unsigned \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half(const unsigned long val) {
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(unsigned long) == sizeof(unsigned long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (default: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            __x = __ull2half_rn(static_cast<unsigned long long>(val)).__x;
        } else {
            __x = __uint2half_rn(static_cast<unsigned int>(val)).__x;
        }
    }

    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Construct \p __half from \p long \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half(const long long val) { __x = __ll2half_rn(val).__x; }
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Construct \p __half from \p unsigned \p long \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half(const unsigned long long val) { __x = __ull2half_rn(val).__x; }

    /* Allow automatic casts to supported built-in types, matching all that are permitted with float */

    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p signed \p char data type.
     * Using round-toward-zero rounding mode.
     * 
     * \see __half2char_rz(__half) for further details.
     */
    __CUDA_HOSTDEVICE__ operator signed char() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p unsigned \p char data type.
     * Using round-toward-zero rounding mode.
     * 
     * \see __half2uchar_rz(__half) for further details.
     */
    __CUDA_HOSTDEVICE__ operator unsigned char() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to an implementation defined \p char data type.
     * Using round-toward-zero rounding mode.
     * 
     * Detects signedness of the \p char type and proceeds accordingly, see
     * further details in __half2char_rz(__half) and __half2uchar_rz(__half).
     */
    __CUDA_HOSTDEVICE__ operator char() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p short data type.
     * Using round-toward-zero rounding mode.
     * 
     * \see __half2short_rz(__half) for further details.
     */
    __CUDA_HOSTDEVICE__ operator short() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p unsigned \p short data type.
     * Using round-toward-zero rounding mode.
     * 
     * \see __half2ushort_rz(__half) for further details.
     */
    __CUDA_HOSTDEVICE__ operator unsigned short() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p int data type.
     * Using round-toward-zero rounding mode.
     * 
     * \see __half2int_rz(__half) for further details.
     */
    __CUDA_HOSTDEVICE__ operator int() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p unsigned \p int data type.
     * Using round-toward-zero rounding mode.
     * 
     * \see __half2uint_rz(__half) for further details.
     */
    __CUDA_HOSTDEVICE__ operator unsigned int() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p long data type.
     * Using round-toward-zero rounding mode.
     *
     * Detects size of the \p long type and proceeds accordingly, see
     * further details in __half2int_rz(__half) and __half2ll_rz(__half).
     */
    __CUDA_HOSTDEVICE__ operator long() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p unsigned \p long data type.
     * Using round-toward-zero rounding mode.
     *
     * Detects size of the \p unsigned \p long type and proceeds
     * accordingly, see further details in __half2uint_rz(__half) and __half2ull_rz(__half).
     */
    __CUDA_HOSTDEVICE__ operator unsigned long() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p long \p long data type.
     * Using round-toward-zero rounding mode.
     * 
     * \see __half2ll_rz(__half) for further details.
     */
    __CUDA_HOSTDEVICE__ operator long long() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p unsigned \p long \p long data type.
     * Using round-toward-zero rounding mode.
     * 
     * \see __half2ull_rz(__half) for further details.
     */
    __CUDA_HOSTDEVICE__ operator unsigned long long() const;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Type cast from \p short assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half &operator=(const short val);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Type cast from \p unsigned \p short assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half &operator=(const unsigned short val);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Type cast from \p int assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half &operator=(const int val);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Type cast from \p unsigned \p int assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half &operator=(const unsigned int val);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Type cast from \p long \p long assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half &operator=(const long long val);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Type cast from \p unsigned \p long \p long assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __half &operator=(const unsigned long long val);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p bool data type.
     * +0 and -0 inputs convert to \p false.
     * Non-zero inputs convert to \p true.
     */
    __CUDA_HOSTDEVICE__ __CUDA_FP16_CONSTEXPR__ operator bool() const { return (__x & 0x7FFFU) != 0U; }
#endif /* #if !(defined __CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__) || (defined __CUDACC__) */
#endif /* !defined(__CUDA_NO_HALF_CONVERSIONS__) */
};

#if !defined(__CUDA_NO_HALF_OPERATORS__)
/* Some basic arithmetic operations expected of a built-in */

/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half addition operation.
 * \see __hadd(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator+(const __half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half subtraction operation.
 * \see __hsub(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator-(const __half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half multiplication operation.
 * \see __hmul(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator*(const __half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half division operation.
 * \see __hdiv(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator/(const __half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half compound assignment with addition operation.
 * \see __hadd(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator+=(__half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half compound assignment with subtraction operation.
 * \see __hsub(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator-=(__half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half compound assignment with multiplication operation.
 * \see __hmul(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator*=(__half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half compound assignment with division operation.
 * \see __hdiv(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator/=(__half &lh, const __half &rh);
/* Note for increment and decrement we use the raw value 0x3C00U equating to half(1.0F), to avoid the extra conversion */
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half prefix increment operation.
 * \see __hadd(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator++(__half &h);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half prefix decrement operation.
 * \see __hsub(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator--(__half &h);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half postfix increment operation.
 * \see __hadd(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half  operator++(__half &h, const int ignored);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Performs \p half postfix decrement operation.
 * \see __hsub(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half  operator--(__half &h, const int ignored);

/* Unary plus and inverse operators */
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Implements \p half unary plus operator, returns input value.
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator+(const __half &h);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * Implements \p half unary minus operator.
 * \see __hneg(__half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator-(const __half &h);
/* Some basic comparison operations to make it look like a built-in */
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * Performs \p half ordered compare equal operation.
 * \see __heq(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator==(const __half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * Performs \p half unordered compare not-equal operation.
 * \see __hneu(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator!=(const __half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * Performs \p half ordered greater-than compare operation.
 * \see __hgt(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator> (const __half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * Performs \p half ordered less-than compare operation.
 * \see __hlt(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator< (const __half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * Performs \p half ordered greater-or-equal compare operation.
 * \see __hge(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator>=(const __half &lh, const __half &rh);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * Performs \p half ordered less-or-equal compare operation.
 * \see __hle(__half, __half)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator<=(const __half &lh, const __half &rh);
#endif /* !defined(__CUDA_NO_HALF_OPERATORS__) */

/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief __half2 data type
 * \details This structure implements the datatype for storing two 
 * half-precision floating-point numbers. 
 * The structure implements assignment, arithmetic and comparison
 * operators, and type conversions. 
 * 
 * - NOTE: __half2 is visible to non-nvcc host compilers
 */
struct __CUDA_ALIGN__(4) __half2 {
    /**
     * Storage field holding lower \p __half part.
     */
    __half x;
    /**
     * Storage field holding upper \p __half part.
     */
    __half y;

    // All construct/copy/assign/move
public:
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * \brief Constructor by default.
     * \details Emtpy default constructor, result is uninitialized.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP16)
    __half2() = default;
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Move constructor, available for \p C++11 and later dialects
     */
    __CUDA_HOSTDEVICE__ __half2(const __half2 &&src) {
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __HALF2_TO_UI(*this) = std::move(__HALF2_TO_CUI(src));
,
    this->x = src.x;
    this->y = src.y;
)
}
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Move assignment operator, available for \p C++11 and later dialects
     */
    __CUDA_HOSTDEVICE__ __half2 &operator=(const __half2 &&src);
#else
    __CUDA_HOSTDEVICE__ __half2() { }
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */

    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Constructor from two \p __half variables
     */
    __CUDA_HOSTDEVICE__ __CUDA_FP16_CONSTEXPR__ __half2(const __half &a, const __half &b) : x(a), y(b) { }
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Copy constructor
     */
    __CUDA_HOSTDEVICE__ __half2(const __half2 &src) {
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __HALF2_TO_UI(*this) = __HALF2_TO_CUI(src);
,
    this->x = src.x;
    this->y = src.y;
)
}    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Copy assignment operator
     */
    __CUDA_HOSTDEVICE__ __half2 &operator=(const __half2 &src);

    /* Convert to/from __half2_raw */
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Constructor from \p __half2_raw
     */
    __CUDA_HOSTDEVICE__ __half2(const __half2_raw &h2r ) {
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __HALF2_TO_UI(*this) = __HALF2_TO_CUI(h2r);
,
    __half_raw tr;
    tr.x = h2r.x;
    this->x = static_cast<__half>(tr);
    tr.x = h2r.y;
    this->y = static_cast<__half>(tr);
)
}
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Assignment operator from \p __half2_raw
     */
    __CUDA_HOSTDEVICE__ __half2 &operator=(const __half2_raw &h2r);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p __half2_raw
     */
    __CUDA_HOSTDEVICE__ operator __half2_raw() const;
};

#if !defined(__CUDA_NO_HALF2_OPERATORS__)
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half addition operation.
 * \see __hadd2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator+(const __half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half subtraction operation.
 * \see __hsub2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator-(const __half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half multiplication operation.
 * \see __hmul2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator*(const __half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half division operation.
 * \see __h2div(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator/(const __half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half compound assignment with addition operation.
 * \see __hadd2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2& operator+=(__half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half compound assignment with subtraction operation.
 * \see __hsub2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2& operator-=(__half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half compound assignment with multiplication operation.
 * \see __hmul2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2& operator*=(__half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half compound assignment with division operation.
 * \see __h2div(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2& operator/=(__half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half prefix increment operation.
 * \see __hadd2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 &operator++(__half2 &h);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half prefix decrement operation.
 * \see __hsub2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 &operator--(__half2 &h);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half postfix increment operation.
 * \see __hadd2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2  operator++(__half2 &h, const int ignored);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Performs packed \p half postfix decrement operation.
 * \see __hsub2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2  operator--(__half2 &h, const int ignored);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Implements packed \p half unary plus operator, returns input value.
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator+(const __half2 &h);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * Implements packed \p half unary minus operator.
 * \see __hneg2(__half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator-(const __half2 &h);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * Performs packed \p half ordered compare equal operation.
 * \see __hbeq2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator==(const __half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * Performs packed \p half unordered compare not-equal operation.
 * \see __hbneu2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator!=(const __half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * Performs packed \p half ordered greater-than compare operation.
 * \see __hbgt2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator>(const __half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * Performs packed \p half ordered less-than compare operation.
 * \see __hblt2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator<(const __half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * Performs packed \p half ordered greater-or-equal compare operation.
 * \see __hbge2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator>=(const __half2 &lh, const __half2 &rh);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * Performs packed \p half ordered less-or-equal compare operation.
 * \see __hble2(__half2, __half2)
 */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator<=(const __half2 &lh, const __half2 &rh);

#endif /* !defined(__CUDA_NO_HALF2_OPERATORS__) */
#endif /* defined(__cplusplus) */

#if (defined(__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__) || \
    !(defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3)))))

/* Note the .hpp file is included to capture the "half" & "half2" built-in function definitions. For NVRTC, the built-in
   function definitions are compiled at NVRTC library build-time and are available through the NVRTC built-ins library at
   link time.
*/
#include "cuda_fp16.hpp"
#endif /* (defined(__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__) || \
          !(defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3))))) */

/* Define first-class types "half" and "half2", unless user specifies otherwise via "#define CUDA_NO_HALF" */
/* C cannot ever have these types defined here, because __half and __half2 are C++ classes */
#if defined(__cplusplus) && !defined(CUDA_NO_HALF)
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief This datatype is meant to be the first-class or fundamental
 * implementation of the half-precision numbers format.
 * 
 * \details Should be implemented in the compiler in the future.
 * Current implementation is a simple typedef to a respective
 * user-level type with underscores.
 */
typedef __half half;

/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief This datatype is meant to be the first-class or fundamental
 * implementation of type for pairs of half-precision numbers.
 * 
 * \details Should be implemented in the compiler in the future.
 * Current implementation is a simple typedef to a respective
 * user-level type with underscores.
 */
typedef __half2 half2;
// for consistency with __nv_bfloat16

/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief This datatype is an \p __nv_ prefixed alias
 */
typedef __half      __nv_half;
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief This datatype is an \p __nv_ prefixed alias
 */
typedef __half2     __nv_half2;
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief This datatype is an \p __nv_ prefixed alias
 */
typedef __half_raw  __nv_half_raw;
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief This datatype is an \p __nv_ prefixed alias
 */
typedef __half2_raw __nv_half2_raw;
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief This datatype is an \p nv_ prefixed alias
 */
typedef __half        nv_half;
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief This datatype is an \p nv_ prefixed alias
 */
typedef __half2       nv_half2;
#endif /* defined(__cplusplus) && !defined(CUDA_NO_HALF) */

#undef __CUDA_FP16_DECL__
#undef __CUDA_HOSTDEVICE_FP16_DECL__
#undef __CUDA_HOSTDEVICE__
#undef __CUDA_FP16_INLINE__
#undef __CUDA_FP16_FORCEINLINE__
#undef ___CUDA_FP16_STRINGIFY_INNERMOST
#undef __CUDA_FP16_STRINGIFY

#endif /* end of include guard: __CUDA_FP16_H__ */
