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

#if !defined(__CUDA_FP16_HPP__)
#define __CUDA_FP16_HPP__

#if !defined(__CUDA_FP16_H__)
#error "Do not include this file directly. Instead, include cuda_fp16.h."
#endif

#if !defined(IF_DEVICE_OR_CUDACC)
#if defined(__CUDACC__)
    #define IF_DEVICE_OR_CUDACC(d, c, f) NV_IF_ELSE_TARGET(NV_IS_DEVICE, d, c)
#else
    #define IF_DEVICE_OR_CUDACC(d, c, f) NV_IF_ELSE_TARGET(NV_IS_DEVICE, d, f)
#endif
#endif

/* Macros for half & half2 binary arithmetic */
#define __BINARY_OP_HALF_MACRO(name) /* do */ {\
   __half val; \
   asm( "{" __CUDA_FP16_STRINGIFY(name) ".f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); \
   return val; \
} /* while(0) */
#define __BINARY_OP_HALF2_MACRO(name) /* do */ {\
   __half2 val; \
   asm( "{" __CUDA_FP16_STRINGIFY(name) ".f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); \
   return val; \
} /* while(0) */
#define __TERNARY_OP_HALF_MACRO(name) /* do */ {\
   __half val; \
   asm( "{" __CUDA_FP16_STRINGIFY(name) ".f16 %0,%1,%2,%3;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b)),"h"(__HALF_TO_CUS(c))); \
   return val; \
} /* while(0) */
#define __TERNARY_OP_HALF2_MACRO(name) /* do */ {\
   __half2 val; \
   asm( "{" __CUDA_FP16_STRINGIFY(name) ".f16x2 %0,%1,%2,%3;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b)),"r"(__HALF2_TO_CUI(c))); \
   return val; \
} /* while(0) */

/* All other definitions in this file are only visible to C++ compilers */
#if defined(__cplusplus)

/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF_CONSTANTS
 * \brief Defines floating-point positive infinity value for the \p half data type
 */
#define CUDART_INF_FP16            __ushort_as_half((unsigned short)0x7C00U)
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF_CONSTANTS
 * \brief Defines canonical NaN value for the \p half data type
 */
#define CUDART_NAN_FP16            __ushort_as_half((unsigned short)0x7FFFU)
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF_CONSTANTS
 * \brief Defines a minimum representable (denormalized) value for the \p half data type
 */
#define CUDART_MIN_DENORM_FP16     __ushort_as_half((unsigned short)0x0001U)
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF_CONSTANTS
 * \brief Defines a maximum representable value for the \p half data type
 */
#define CUDART_MAX_NORMAL_FP16     __ushort_as_half((unsigned short)0x7BFFU)
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF_CONSTANTS
 * \brief Defines a negative zero value for the \p half data type
 */
#define CUDART_NEG_ZERO_FP16       __ushort_as_half((unsigned short)0x8000U)
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF_CONSTANTS
 * \brief Defines a positive zero value for the \p half data type
 */
#define CUDART_ZERO_FP16           __ushort_as_half((unsigned short)0x0000U)
/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF_CONSTANTS
 * \brief Defines a value of 1.0 for the \p half data type
 */
#define CUDART_ONE_FP16            __ushort_as_half((unsigned short)0x3C00U)

#if !(defined __DOXYGEN_ONLY__)

__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half &__half::operator=(const __half_raw &hr) { __x = hr.x; return *this; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ volatile __half &__half::operator=(const __half_raw &hr) volatile { __x = hr.x; return *this; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ volatile __half &__half::operator=(const volatile __half_raw &hr) volatile { __x = hr.x; return *this; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator __half_raw() const { __half_raw ret; ret.x = __x; return ret; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator __half_raw() const volatile { __half_raw ret; ret.x = __x; return ret; }
#if !defined(__CUDA_NO_HALF_CONVERSIONS__)
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator float() const { return __half2float(*this); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half &__half::operator=(const float f) { __x = __float2half(f).__x; return *this; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half &__half::operator=(const double f) { __x = __double2half(f).__x; return *this; }
#if !(defined __CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__) || (defined __CUDACC__)
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator signed char() const { return __half2char_rz(*this); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator unsigned char() const { return __half2uchar_rz(*this); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator char() const {
    char value;
    /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (push)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
    if (((char)-1) < (char)0)
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (pop)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
    {
        value = static_cast<char>(__half2char_rz(*this));
    }
    else
    {
        value = static_cast<char>(__half2uchar_rz(*this));
    }
    return value;
}
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator short() const { return __half2short_rz(*this); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator unsigned short() const { return __half2ushort_rz(*this); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator int() const { return __half2int_rz(*this); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator unsigned int() const { return __half2uint_rz(*this); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator long() const {
    long retval;
    /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (push)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
    if (sizeof(long) == sizeof(long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (pop)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
    {
        retval = static_cast<long>(__half2ll_rz(*this));
    }
    else
    {
        retval = static_cast<long>(__half2int_rz(*this));
    }
    return retval;
}
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator unsigned long() const {
    unsigned long retval;
    /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (push)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
    if (sizeof(unsigned long) == sizeof(unsigned long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (pop)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
    {
        retval = static_cast<unsigned long>(__half2ull_rz(*this));
    }
    else
    {
        retval = static_cast<unsigned long>(__half2uint_rz(*this));
    }
    return retval;
}
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator long long() const { return __half2ll_rz(*this); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half::operator unsigned long long() const { return __half2ull_rz(*this); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half &__half::operator=(const short val) { __x = __short2half_rn(val).__x; return *this; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half &__half::operator=(const unsigned short val) { __x = __ushort2half_rn(val).__x; return *this; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half &__half::operator=(const int val) { __x = __int2half_rn(val).__x; return *this; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half &__half::operator=(const unsigned int val) { __x = __uint2half_rn(val).__x; return *this; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half &__half::operator=(const long long val) { __x = __ll2half_rn(val).__x; return *this; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half &__half::operator=(const unsigned long long val) { __x = __ull2half_rn(val).__x; return *this; }

#endif /* #if !(defined __CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__) || (defined __CUDACC__) */
#endif /* !defined(__CUDA_NO_HALF_CONVERSIONS__) */
#if !defined(__CUDA_NO_HALF_OPERATORS__)
/* Some basic arithmetic operations expected of a builtin */
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator+(const __half &lh, const __half &rh) { return __hadd(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator-(const __half &lh, const __half &rh) { return __hsub(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator*(const __half &lh, const __half &rh) { return __hmul(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator/(const __half &lh, const __half &rh) { return __hdiv(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator+=(__half &lh, const __half &rh) { lh = __hadd(lh, rh); return lh; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator-=(__half &lh, const __half &rh) { lh = __hsub(lh, rh); return lh; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator*=(__half &lh, const __half &rh) { lh = __hmul(lh, rh); return lh; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator/=(__half &lh, const __half &rh) { lh = __hdiv(lh, rh); return lh; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator++(__half &h)      { __half_raw one; one.x = 0x3C00U; h += one; return h; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half &operator--(__half &h)      { __half_raw one; one.x = 0x3C00U; h -= one; return h; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half  operator++(__half &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __half ret = h;
    __half_raw one;
    one.x = 0x3C00U;
    h += one;
    return ret;
}
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half  operator--(__half &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __half ret = h;
    __half_raw one;
    one.x = 0x3C00U;
    h -= one;
    return ret;
}
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator+(const __half &h) { return h; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half operator-(const __half &h) { return __hneg(h); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator==(const __half &lh, const __half &rh) { return __heq(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator!=(const __half &lh, const __half &rh) { return __hneu(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator> (const __half &lh, const __half &rh) { return __hgt(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator< (const __half &lh, const __half &rh) { return __hlt(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator>=(const __half &lh, const __half &rh) { return __hge(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator<=(const __half &lh, const __half &rh) { return __hle(lh, rh); }
#endif /* !defined(__CUDA_NO_HALF_OPERATORS__) */
#if defined(__CPP_VERSION_AT_LEAST_11_FP16)
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half2 &__half2::operator=(const __half2 &&src) {
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __HALF2_TO_UI(*this) = std::move(__HALF2_TO_CUI(src));
,
    this->x = src.x;
    this->y = src.y;
)
    return *this;
}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half2 &__half2::operator=(const __half2 &src) {
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __HALF2_TO_UI(*this) = __HALF2_TO_CUI(src);
,
    this->x = src.x;
    this->y = src.y;
)
    return *this;
}
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half2 &__half2::operator=(const __half2_raw &h2r) {
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __HALF2_TO_UI(*this) = __HALF2_TO_CUI(h2r);
,
    __half_raw tr;
    tr.x = h2r.x;
    this->x = static_cast<__half>(tr);
    tr.x = h2r.y;
    this->y = static_cast<__half>(tr);
)
    return *this;
}
__CUDA_HOSTDEVICE__ __CUDA_FP16_INLINE__ __half2::operator __half2_raw() const {
    __half2_raw ret;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    ret.x = 0U;
    ret.y = 0U;
    __HALF2_TO_UI(ret) = __HALF2_TO_CUI(*this);
,
    ret.x = static_cast<__half_raw>(this->x).x;
    ret.y = static_cast<__half_raw>(this->y).x;
)
    return ret;
}
#if !defined(__CUDA_NO_HALF2_OPERATORS__)
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator+(const __half2 &lh, const __half2 &rh) { return __hadd2(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator-(const __half2 &lh, const __half2 &rh) { return __hsub2(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator*(const __half2 &lh, const __half2 &rh) { return __hmul2(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator/(const __half2 &lh, const __half2 &rh) { return __h2div(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2& operator+=(__half2 &lh, const __half2 &rh) { lh = __hadd2(lh, rh); return lh; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2& operator-=(__half2 &lh, const __half2 &rh) { lh = __hsub2(lh, rh); return lh; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2& operator*=(__half2 &lh, const __half2 &rh) { lh = __hmul2(lh, rh); return lh; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2& operator/=(__half2 &lh, const __half2 &rh) { lh = __h2div(lh, rh); return lh; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 &operator++(__half2 &h)      { __half2_raw one; one.x = 0x3C00U; one.y = 0x3C00U; h = __hadd2(h, one); return h; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 &operator--(__half2 &h)      { __half2_raw one; one.x = 0x3C00U; one.y = 0x3C00U; h = __hsub2(h, one); return h; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2  operator++(__half2 &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __half2 ret = h;
    __half2_raw one;
    one.x = 0x3C00U;
    one.y = 0x3C00U;
    h = __hadd2(h, one);
    return ret;
}
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2  operator--(__half2 &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __half2 ret = h;
    __half2_raw one;
    one.x = 0x3C00U;
    one.y = 0x3C00U;
    h = __hsub2(h, one);
    return ret;
}
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator+(const __half2 &h) { return h; }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ __half2 operator-(const __half2 &h) { return __hneg2(h); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator==(const __half2 &lh, const __half2 &rh) { return __hbeq2(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator!=(const __half2 &lh, const __half2 &rh) { return __hbneu2(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator>(const __half2 &lh, const __half2 &rh) { return __hbgt2(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator<(const __half2 &lh, const __half2 &rh) { return __hblt2(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator>=(const __half2 &lh, const __half2 &rh) { return __hbge2(lh, rh); }
__CUDA_HOSTDEVICE__ __CUDA_FP16_FORCEINLINE__ bool operator<=(const __half2 &lh, const __half2 &rh) { return __hble2(lh, rh); }
#endif /* !defined(__CUDA_NO_HALF2_OPERATORS__) */

/* Restore warning for multiple assignment operators */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( pop )
#endif /* defined(_MSC_VER) && _MSC_VER >= 1500 */

/* Restore -Weffc++ warnings from here on */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

#undef __CUDA_HOSTDEVICE__
#undef __CUDA_ALIGN__

#ifndef __CUDACC_RTC__  /* no host functions in NVRTC mode */
static inline unsigned short __internal_float2half(const float f, unsigned int &sign, unsigned int &remainder)
{
    unsigned int x;
    unsigned int u;
    unsigned int result;
#if defined(__CUDACC__)
    (void)memcpy(&x, &f, sizeof(f));
#else
    (void)std::memcpy(&x, &f, sizeof(f));
#endif
    u = (x & 0x7fffffffU);
    sign = ((x >> 16U) & 0x8000U);
    // NaN/+Inf/-Inf
    if (u >= 0x7f800000U) {
        remainder = 0U;
        result = ((u == 0x7f800000U) ? (sign | 0x7c00U) : 0x7fffU);
    } else if (u > 0x477fefffU) { // Overflows
        remainder = 0x80000000U;
        result = (sign | 0x7bffU);
    } else if (u >= 0x38800000U) { // Normal numbers
        remainder = u << 19U;
        u -= 0x38000000U;
        result = (sign | (u >> 13U));
    } else if (u < 0x33000001U) { // +0/-0
        remainder = u;
        result = sign;
    } else { // Denormal numbers
        const unsigned int exponent = u >> 23U;
        const unsigned int shift = 0x7eU - exponent;
        unsigned int mantissa = (u & 0x7fffffU);
        mantissa |= 0x800000U;
        remainder = mantissa << (32U - shift);
        result = (sign | (mantissa >> shift));
        result &= 0x0000FFFFU;
    }
    return static_cast<unsigned short>(result);
}
#endif  /* #if !defined(__CUDACC_RTC__) */

__CUDA_HOSTDEVICE_FP16_DECL__ __half __double2half(const double a)
{
IF_DEVICE_OR_CUDACC(
    __half val;
    asm("{  cvt.rn.f16.f64 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "d"(a));
    return val;
,
    __half result;
    // Perform rounding to 11 bits of precision, convert value
    // to float and call existing float to half conversion.
    // By pre-rounding to 11 bits we avoid additional rounding
    // in float to half conversion.
    unsigned long long int absa;
    unsigned long long int ua;
    (void)memcpy(&ua, &a, sizeof(a));
    absa = (ua & 0x7fffffffffffffffULL);
    if ((absa >= 0x40f0000000000000ULL) || (absa <= 0x3e60000000000000ULL))
    {
        // |a| >= 2^16 or NaN or |a| <= 2^(-25)
        // double-rounding is not a problem
        result = __float2half(static_cast<float>(a));
    }
    else
    {
        // here 2^(-25) < |a| < 2^16
        // prepare shifter value such that a + shifter
        // done in double precision performs round-to-nearest-even
        // and (a + shifter) - shifter results in a rounded to
        // 11 bits of precision. Shifter needs to have exponent of
        // a plus 53 - 11 = 42 and a leading bit in mantissa to guard
        // against negative values.
        // So need to have |a| capped to avoid overflow in exponent.
        // For inputs that are smaller than half precision minnorm
        // we prepare fixed shifter exponent.
        unsigned long long shifterBits;
        if (absa >= 0x3f10000000000000ULL)
        {   // Here if |a| >= 2^(-14)
            // add 42 to exponent bits
            shifterBits  = (ua & 0x7ff0000000000000ULL) + 0x02A0000000000000ULL;
        }
        else
        {   // 2^(-25) < |a| < 2^(-14), potentially results in denormal
            // set exponent bits to 42 - 14 + bias
            shifterBits = 0x41B0000000000000ULL;
        }
        // set leading mantissa bit to protect against negative inputs
        shifterBits |= 0x0008000000000000ULL;
        double shifter;
        (void)memcpy(&shifter, &shifterBits, sizeof(shifterBits));
        double aShiftRound = a + shifter;

        // Prevent the compiler from optimizing away a + shifter - shifter
        // by doing intermediate memcopy and harmless bitwize operation
        unsigned long long int aShiftRoundBits;
        (void)memcpy(&aShiftRoundBits, &aShiftRound, sizeof(aShiftRound));

        // the value is positive, so this operation doesn't change anything
        aShiftRoundBits &= 0x7fffffffffffffffULL;

        (void)memcpy(&aShiftRound, &aShiftRoundBits, sizeof(aShiftRound));

        result = __float2half(static_cast<float>(aShiftRound - shifter));
    }

    return result;
,
    __half result;
    /*
    // Perform rounding to 11 bits of precision, convert value
    // to float and call existing float to half conversion.
    // By pre-rounding to 11 bits we avoid additional rounding
    // in float to half conversion.
    */
    unsigned long long int absa;
    unsigned long long int ua;
    (void)std::memcpy(&ua, &a, sizeof(a));
    absa = (ua & 0x7fffffffffffffffULL);
    if ((absa >= 0x40f0000000000000ULL) || (absa <= 0x3e60000000000000ULL))
    {
        /*
        // |a| >= 2^16 or NaN or |a| <= 2^(-25)
        // double-rounding is not a problem
        */
        result = __float2half(static_cast<float>(a));
    }
    else
    {
        /*
        // here 2^(-25) < |a| < 2^16
        // prepare shifter value such that a + shifter
        // done in double precision performs round-to-nearest-even
        // and (a + shifter) - shifter results in a rounded to
        // 11 bits of precision. Shifter needs to have exponent of
        // a plus 53 - 11 = 42 and a leading bit in mantissa to guard
        // against negative values.
        // So need to have |a| capped to avoid overflow in exponent.
        // For inputs that are smaller than half precision minnorm
        // we prepare fixed shifter exponent.
        */
        unsigned long long shifterBits;
        if (absa >= 0x3f10000000000000ULL)
        {
            /*
            // Here if |a| >= 2^(-14)
            // add 42 to exponent bits
            */
            shifterBits  = (ua & 0x7ff0000000000000ULL) + 0x02A0000000000000ULL;
        }
        else
        {
            /*
            // 2^(-25) < |a| < 2^(-14), potentially results in denormal
            // set exponent bits to 42 - 14 + bias
            */
            shifterBits = 0x41B0000000000000ULL;
        }
        // set leading mantissa bit to protect against negative inputs
        shifterBits |= 0x0008000000000000ULL;
        double shifter;
        (void)std::memcpy(&shifter, &shifterBits, sizeof(shifterBits));
        double aShiftRound = a + shifter;

        /*
        // Prevent the compiler from optimizing away a + shifter - shifter
        // by doing intermediate memcopy and harmless bitwize operation
        */
        unsigned long long int aShiftRoundBits;
        (void)std::memcpy(&aShiftRoundBits, &aShiftRound, sizeof(aShiftRound));

        // the value is positive, so this operation doesn't change anything
        aShiftRoundBits &= 0x7fffffffffffffffULL;

        (void)std::memcpy(&aShiftRound, &aShiftRoundBits, sizeof(aShiftRound));

        result = __float2half(static_cast<float>(aShiftRound - shifter));
    }

    return result;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half(const float a)
{
    __half val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(a));
,
    __half_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2half(a, sign, remainder);
    if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
        r.x++;
    }
    val = r;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rn(const float a)
{
    __half val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(a));
,
    __half_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2half(a, sign, remainder);
    if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
        r.x++;
    }
    val = r;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rz(const float a)
{
    __half val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{  cvt.rz.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(a));
,
    __half_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2half(a, sign, remainder);
    val = r;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rd(const float a)
{
    __half val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{  cvt.rm.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(a));
,
    __half_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2half(a, sign, remainder);
    if ((remainder != 0U) && (sign != 0U)) {
        r.x++;
    }
    val = r;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_ru(const float a)
{
    __half val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{  cvt.rp.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(a));
,
    __half_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2half(a, sign, remainder);
    if ((remainder != 0U) && (sign == 0U)) {
        r.x++;
    }
    val = r;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __float2half2_rn(const float a)
{
    __half2 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .f16 low;\n"
        "  cvt.rn.f16.f32 low, %1;\n"
        "  mov.b32 %0, {low,low};}\n" : "=r"(__HALF2_TO_UI(val)) : "f"(a));
,
    val = __half2(__float2half_rn(a), __float2half_rn(a));
)
    return val;
}

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ __half2 __internal_device_float2_to_half2_rn(const float a, const float b) {
    __half2 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm("{ cvt.rn.f16x2.f32 %0, %2, %1; }\n"
        : "=r"(__HALF2_TO_UI(val)) : "f"(a), "f"(b));
,
    asm("{.reg .f16 low,high;\n"
        "  cvt.rn.f16.f32 low, %1;\n"
        "  cvt.rn.f16.f32 high, %2;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "f"(a), "f"(b));
)
    return val;
}

#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __floats2half2_rn(const float a, const float b)
{
    __half2 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    val = __internal_device_float2_to_half2_rn(a,b);
,
    val = __half2(__float2half_rn(a), __float2half_rn(b));
)
    return val;
}

#ifndef __CUDACC_RTC__  /* no host functions in NVRTC mode */
static inline float __internal_half2float(const unsigned short h)
{
    unsigned int sign = ((static_cast<unsigned int>(h) >> 15U) & 1U);
    unsigned int exponent = ((static_cast<unsigned int>(h) >> 10U) & 0x1fU);
    unsigned int mantissa = ((static_cast<unsigned int>(h) & 0x3ffU) << 13U);
    float f;
    if (exponent == 0x1fU) { /* NaN or Inf */
        /* discard sign of a NaN */
        sign = ((mantissa != 0U) ? (sign >> 1U) : sign);
        mantissa = ((mantissa != 0U) ? 0x7fffffU : 0U);
        exponent = 0xffU;
    } else if (exponent == 0U) { /* Denorm or Zero */
        if (mantissa != 0U) {
            unsigned int msb;
            exponent = 0x71U;
            do {
                msb = (mantissa & 0x400000U);
                mantissa <<= 1U; /* normalize */
                --exponent;
            } while (msb == 0U);
            mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70U;
    }
    const unsigned int u = ((sign << 31U) | (exponent << 23U) | mantissa);
#if defined(__CUDACC__)
    (void)memcpy(&f, &u, sizeof(u));
#else
    (void)std::memcpy(&f, &u, sizeof(u));
#endif
    return f;
}
#endif  /* !defined(__CUDACC_RTC__) */

__CUDA_HOSTDEVICE_FP16_DECL__ float __half2float(const __half a)
{
    float val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(a)));
,
    val = __internal_half2float(static_cast<__half_raw>(a).x);
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ float __low2float(const __half2 a)
{
    float val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, low;}\n" : "=f"(val) : "r"(__HALF2_TO_CUI(a)));
,
    val = __internal_half2float(static_cast<__half2_raw>(a).x);
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ float __high2float(const __half2 a)
{
    float val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, high;}\n" : "=f"(val) : "r"(__HALF2_TO_CUI(a)));
,
    val = __internal_half2float(static_cast<__half2_raw>(a).y);
)
    return val;
}

__CUDA_HOSTDEVICE_FP16_DECL__ signed char __half2char_rz(const __half h)
{
    signed char i;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    unsigned int tmp;
    asm("cvt.rzi.s8.f16 %0, %1;" : "=r"(tmp) : "h"(__HALF_TO_CUS(h)));
    const unsigned char u = static_cast<unsigned char>(tmp);
    i = static_cast<signed char>(u);
,
    const float f = __half2float(h);
    const signed char max_val = (signed char)0x7fU;
    const signed char min_val = (signed char)0x80U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<signed char>(f);
    }
)
    return i;
}

__CUDA_HOSTDEVICE_FP16_DECL__ unsigned char __half2uchar_rz(const __half h)
{
    unsigned char i;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    unsigned int tmp;
    asm("cvt.rzi.u8.f16 %0, %1;" : "=r"(tmp) : "h"(__HALF_TO_CUS(h)));
    i = static_cast<unsigned char>(tmp);
,
    const float f = __half2float(h);
    const unsigned char max_val = 0xffU;
    const unsigned char min_val = 0U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0U;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<unsigned char>(f);
    }
)
    return i;
}

__CUDA_HOSTDEVICE_FP16_DECL__ short int __half2short_rz(const __half h)
{
    short int i;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rzi.s16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_CUS(h)));
,
    const float f = __half2float(h);
    const short int max_val = (short int)0x7fffU;
    const short int min_val = (short int)0x8000U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<short int>(f);
    }
)
    return i;
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned short int __half2ushort_rz(const __half h)
{
    unsigned short int i;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rzi.u16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_CUS(h)));
,
    const float f = __half2float(h);
    const unsigned short int max_val = 0xffffU;
    const unsigned short int min_val = 0U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0U;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<unsigned short int>(f);
    }
)
    return i;
}
__CUDA_HOSTDEVICE_FP16_DECL__ int __half2int_rz(const __half h)
{
    int i;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rzi.s32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_CUS(h)));
,
    const float f = __half2float(h);
    const int max_val = (int)0x7fffffffU;
    const int min_val = (int)0x80000000U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<int>(f);
    }
)
    return i;
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __half2uint_rz(const __half h)
{
    unsigned int i;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rzi.u32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_CUS(h)));
,
    const float f = __half2float(h);
    const unsigned int max_val = 0xffffffffU;
    const unsigned int min_val = 0U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0U;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<unsigned int>(f);
    }
)
    return i;
}
__CUDA_HOSTDEVICE_FP16_DECL__ long long int __half2ll_rz(const __half h)
{
    long long int i;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rzi.s64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_CUS(h)));
,
    const float f = __half2float(h);
    const long long int max_val = (long long int)0x7fffffffffffffffULL;
    const long long int min_val = (long long int)0x8000000000000000ULL;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = min_val;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<long long int>(f);
    }
)
    return i;
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned long long int __half2ull_rz(const __half h)
{
    unsigned long long int i;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rzi.u64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_CUS(h)));
,
    const float f = __half2float(h);
    const unsigned long long int max_val = 0xffffffffffffffffULL;
    const unsigned long long int min_val = 0ULL;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0x8000000000000000ULL;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<unsigned long long int>(f);
    }
)
    return i;
}
/* CUDA vector-types compatible vector creation function (note returns __half2, not half2) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 make_half2(const __half x, const __half y)
{
    __half2 t; t.x = x; t.y = y; return t;
}


/* Definitions of intrinsics */
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __float22half2_rn(const float2 a)
{
    const __half2 val = __floats2half2_rn(a.x, a.y);
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ float2 __half22float2(const __half2 a)
{
    float hi_float;
    float lo_float;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, low;}\n" : "=f"(lo_float) : "r"(__HALF2_TO_CUI(a)));

    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, high;}\n" : "=f"(hi_float) : "r"(__HALF2_TO_CUI(a)));
,
    lo_float = __internal_half2float(((__half2_raw)a).x);
    hi_float = __internal_half2float(((__half2_raw)a).y);
)
    return make_float2(lo_float, hi_float);
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ int __half2int_rn(const __half h)
{
    int i;
    asm("cvt.rni.s32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ int __half2int_rd(const __half h)
{
    int i;
    asm("cvt.rmi.s32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ int __half2int_ru(const __half h)
{
    int i;
    asm("cvt.rpi.s32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half __int2half_rn(const int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rn.f16.s32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
,
    // double-rounding is not a problem here: if integer
    // has more than 24 bits, it is already too large to
    // be represented in half precision, and result will
    // be infinity.
    const float  f = static_cast<float>(i);
                 h = __float2half_rn(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __int2half_rz(const int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rz.f16.s32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rz(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __int2half_rd(const int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rm.f16.s32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rd(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __int2half_ru(const int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rp.f16.s32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_ru(f);
)
    return h;
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ short int __half2short_rn(const __half h)
{
    short int i;
    asm("cvt.rni.s16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ short int __half2short_rd(const __half h)
{
    short int i;
    asm("cvt.rmi.s16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ short int __half2short_ru(const __half h)
{
    short int i;
    asm("cvt.rpi.s16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half __short2half_rn(const short int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rn.f16.s16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rn(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __short2half_rz(const short int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rz.f16.s16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rz(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __short2half_rd(const short int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rm.f16.s16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rd(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __short2half_ru(const short int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rp.f16.s16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_ru(f);
)
    return h;
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ unsigned int __half2uint_rn(const __half h)
{
    unsigned int i;
    asm("cvt.rni.u32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned int __half2uint_rd(const __half h)
{
    unsigned int i;
    asm("cvt.rmi.u32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned int __half2uint_ru(const __half h)
{
    unsigned int i;
    asm("cvt.rpi.u32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half __uint2half_rn(const unsigned int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rn.f16.u32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
,
    // double-rounding is not a problem here: if integer
    // has more than 24 bits, it is already too large to
    // be represented in half precision, and result will
    // be infinity.
    const float  f = static_cast<float>(i);
                 h = __float2half_rn(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __uint2half_rz(const unsigned int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rz.f16.u32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rz(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __uint2half_rd(const unsigned int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rm.f16.u32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rd(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __uint2half_ru(const unsigned int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rp.f16.u32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_ru(f);
)
    return h;
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rn(const __half h)
{
    unsigned short int i;
    asm("cvt.rni.u16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rd(const __half h)
{
    unsigned short int i;
    asm("cvt.rmi.u16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned short int __half2ushort_ru(const __half h)
{
    unsigned short int i;
    asm("cvt.rpi.u16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort2half_rn(const unsigned short int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rn.f16.u16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rn(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort2half_rz(const unsigned short int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rz.f16.u16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rz(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort2half_rd(const unsigned short int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rm.f16.u16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rd(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort2half_ru(const unsigned short int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rp.f16.u16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_ru(f);
)
    return h;
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rn(const __half h)
{
    unsigned long long int i;
    asm("cvt.rni.u64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rd(const __half h)
{
    unsigned long long int i;
    asm("cvt.rmi.u64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned long long int __half2ull_ru(const __half h)
{
    unsigned long long int i;
    asm("cvt.rpi.u64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ull2half_rn(const unsigned long long int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rn.f16.u64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
,
    // double-rounding is not a problem here: if integer
    // has more than 24 bits, it is already too large to
    // be represented in half precision, and result will
    // be infinity.
    const float  f = static_cast<float>(i);
                 h = __float2half_rn(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ull2half_rz(const unsigned long long int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rz.f16.u64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rz(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ull2half_rd(const unsigned long long int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rm.f16.u64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rd(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ull2half_ru(const unsigned long long int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rp.f16.u64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_ru(f);
)
    return h;
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ long long int __half2ll_rn(const __half h)
{
    long long int i;
    asm("cvt.rni.s64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ long long int __half2ll_rd(const __half h)
{
    long long int i;
    asm("cvt.rmi.s64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
__CUDA_FP16_DECL__ long long int __half2ll_ru(const __half h)
{
    long long int i;
    asm("cvt.rpi.s64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_CUS(h)));
    return i;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ll2half_rn(const long long int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rn.f16.s64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
,
    // double-rounding is not a problem here: if integer
    // has more than 24 bits, it is already too large to
    // be represented in half precision, and result will
    // be infinity.
    const float  f = static_cast<float>(i);
                 h = __float2half_rn(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ll2half_rz(const long long int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rz.f16.s64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rz(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ll2half_rd(const long long int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rm.f16.s64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_rd(f);
)
    return h;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ll2half_ru(const long long int i)
{
    __half h;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rp.f16.s64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
,
    const float  f = static_cast<float>(i);
                 h = __float2half_ru(f);
)
    return h;
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ __half htrunc(const __half h)
{
    __half r;
    asm("cvt.rzi.f16.f16 %0, %1;" : "=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(h)));
    return r;
}
__CUDA_FP16_DECL__ __half hceil(const __half h)
{
    __half r;
    asm("cvt.rpi.f16.f16 %0, %1;" : "=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(h)));
    return r;
}
__CUDA_FP16_DECL__ __half hfloor(const __half h)
{
    __half r;
    asm("cvt.rmi.f16.f16 %0, %1;" : "=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(h)));
    return r;
}
__CUDA_FP16_DECL__ __half hrint(const __half h)
{
    __half r;
    asm("cvt.rni.f16.f16 %0, %1;" : "=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(h)));
    return r;
}

__CUDA_FP16_DECL__ __half2 h2trunc(const __half2 h)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rzi.f16.f16 low, low;\n"
        "  cvt.rzi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2ceil(const __half2 h)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rpi.f16.f16 low, low;\n"
        "  cvt.rpi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2floor(const __half2 h)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rmi.f16.f16 low, low;\n"
        "  cvt.rmi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2rint(const __half2 h)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rni.f16.f16 low, low;\n"
        "  cvt.rni.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(h)));
    return val;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __lows2half2(const __half2 a, const __half2 b)
{
    __half2 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
        "  mov.b32 {alow,ahigh}, %1;\n"
        "  mov.b32 {blow,bhigh}, %2;\n"
        "  mov.b32 %0, {alow,blow};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)), "r"(__HALF2_TO_CUI(b)));
,
    val.x = a.x;
    val.y = b.x;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __highs2half2(const __half2 a, const __half2 b)
{
    __half2 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
        "  mov.b32 {alow,ahigh}, %1;\n"
        "  mov.b32 {blow,bhigh}, %2;\n"
        "  mov.b32 %0, {ahigh,bhigh};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)), "r"(__HALF2_TO_CUI(b)));
,
    val.x = a.y;
    val.y = b.y;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __low2half(const __half2 a)
{
    __half ret;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .f16 low,high;\n"
        " mov.b32 {low,high}, %1;\n"
        " mov.b16 %0, low;}" : "=h"(__HALF_TO_US(ret)) : "r"(__HALF2_TO_CUI(a)));
,
    ret = a.x;
)
    return ret;
}
__CUDA_HOSTDEVICE_FP16_DECL__ int __hisinf(const __half a)
{
    int retval;
    const __half_raw araw = __half_raw(a);
    if (araw.x == 0xFC00U) {
        retval = -1;
    } else if (araw.x == 0x7C00U) {
        retval = 1;
    } else {
        retval = 0;
    }
    return retval;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __low2half2(const __half2 a)
{
    __half2 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {low,low};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
,
    val.x = a.x;
    val.y = a.x;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __high2half2(const __half2 a)
{
    __half2 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {high,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
,
    val.x = a.y;
    val.y = a.y;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __high2half(const __half2 a)
{
    __half ret;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .f16 low,high;\n"
        " mov.b32 {low,high}, %1;\n"
        " mov.b16 %0, high;}" : "=h"(__HALF_TO_US(ret)) : "r"(__HALF2_TO_CUI(a)));
,
    ret = a.y;
)
    return ret;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __halves2half2(const __half a, const __half b)
{
    __half2 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{  mov.b32 %0, {%1,%2};}\n"
        : "=r"(__HALF2_TO_UI(val)) : "h"(__HALF_TO_CUS(a)), "h"(__HALF_TO_CUS(b)));
,
    val.x = a;
    val.y = b;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __half2half2(const __half a)
{
    __half2 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{  mov.b32 %0, {%1,%1};}\n"
        : "=r"(__HALF2_TO_UI(val)) : "h"(__HALF_TO_CUS(a)));
,
    val.x = a;
    val.y = a;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __lowhigh2highlow(const __half2 a)
{
    __half2 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {high,low};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
,
    val.x = a.y;
    val.y = a.x;
)
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ short int __half_as_short(const __half h)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return static_cast<short int>(__HALF_TO_CUS(h));
,
    return static_cast<short int>(__half_raw(h).x);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned short int __half_as_ushort(const __half h)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __HALF_TO_CUS(h);
,
    return __half_raw(h).x;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __short_as_half(const short int i)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __half h;
    __HALF_TO_US(h) = static_cast<unsigned short int>(i);
    return h;
,
    __half_raw hr;
    hr.x = static_cast<unsigned short int>(i);
    return __half(hr);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort_as_half(const unsigned short int i)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __half h;
    __HALF_TO_US(h) = i;
    return h;
,
    __half_raw hr;
    hr.x = i;
    return __half(hr);)
}

/******************************************************************************
*                             __half arithmetic                             *
******************************************************************************/
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ __half __internal_device_hmax(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __BINARY_OP_HALF_MACRO(max)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    float fr;
    asm("{max.f32 %0,%1,%2;\n}"
        :"=f"(fr) : "f"(fa), "f"(fb));
    const __half hr = __float2half(fr);
    return hr;
)
}
__CUDA_FP16_DECL__ __half __internal_device_hmin(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __BINARY_OP_HALF_MACRO(min)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    float fr;
    asm("{min.f32 %0,%1,%2;\n}"
        :"=f"(fr) : "f"(fa), "f"(fb));
    const __half hr = __float2half(fr);
    return hr;
)
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmax(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hmax(a, b);
,
    __half maxval;

    maxval = (__hge(a, b) || __hisnan(b)) ? a : b;

    if (__hisnan(maxval))
    {
        // if both inputs are NaN, return canonical NaN
        maxval = CUDART_NAN_FP16;
    }
    else if (__heq(a, b))
    {
        // hmax(+0.0, -0.0) = +0.0
        // unsigned compare 0x8000U > 0x0000U
        __half_raw ra = __half_raw(a);
        __half_raw rb = __half_raw(b);
        maxval = (ra.x > rb.x) ? b : a;
    }
    return maxval;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmin(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hmin(a, b);
,
    __half minval;

    minval = (__hle(a, b) || __hisnan(b)) ? a : b;

    if (__hisnan(minval))
    {
        // if both inputs are NaN, return canonical NaN
        minval = CUDART_NAN_FP16;
    }
    else if (__heq(a, b))
    {
        // hmin(+0.0, -0.0) = -0.0
        // unsigned compare 0x8000U > 0x0000U
        __half_raw ra = __half_raw(a);
        __half_raw rb = __half_raw(b);
        minval = (ra.x > rb.x) ? a : b;
    }

    return minval;
)
}


/******************************************************************************
*                            __half2 arithmetic                             *
******************************************************************************/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmax2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __BINARY_OP_HALF2_MACRO(max)
,
    __half2 val;
    val.x = __hmax(a.x, b.x);
    val.y = __hmax(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmin2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __BINARY_OP_HALF2_MACRO(min)
,
    __half2 val;
    val.x = __hmin(a.x, b.x);
    val.y = __hmin(a.y, b.y);
    return val;
)
}

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300) || defined(_NVHPC_CUDA)
/******************************************************************************
*                           __half, __half2 warp shuffle                     *
******************************************************************************/
#define __SHUFFLE_HALF2_MACRO(name) /* do */ {\
   __half2 r; \
   asm volatile ("{" __CUDA_FP16_STRINGIFY(name) " %0,%1,%2,%3;\n}" \
       :"=r"(__HALF2_TO_UI(r)): "r"(__HALF2_TO_CUI(var)), "r"(delta), "r"(c)); \
   return r; \
} /* while(0) */

#define __SHUFFLE_SYNC_HALF2_MACRO(name, var, delta, c, mask) /* do */ {\
   __half2 r; \
   asm volatile ("{" __CUDA_FP16_STRINGIFY(name) " %0,%1,%2,%3,%4;\n}" \
       :"=r"(__HALF2_TO_UI(r)): "r"(__HALF2_TO_CUI(var)), "r"(delta), "r"(c), "r"(mask)); \
   return r; \
} /* while(0) */

#if defined(_NVHPC_CUDA) || !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 700)

__CUDA_FP16_DECL__ __half2 __shfl(const __half2 var, const int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_HALF2_MACRO(shfl.idx.b32)
}
__CUDA_FP16_DECL__ __half2 __shfl_up(const __half2 var, const unsigned int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = (warp_size - static_cast<unsigned>(width)) << 8U;
    __SHUFFLE_HALF2_MACRO(shfl.up.b32)
}
__CUDA_FP16_DECL__ __half2 __shfl_down(const __half2 var, const unsigned int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_HALF2_MACRO(shfl.down.b32)
}
__CUDA_FP16_DECL__ __half2 __shfl_xor(const __half2 var, const int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_HALF2_MACRO(shfl.bfly.b32)
}

#endif /* defined(_NVHPC_CUDA) || !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 700) */

__CUDA_FP16_DECL__ __half2 __shfl_sync(const unsigned int mask, const __half2 var, const int srcLane, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.idx.b32, var, srcLane, c, mask)
}
__CUDA_FP16_DECL__ __half2 __shfl_up_sync(const unsigned int mask, const __half2 var, const unsigned int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = (warp_size - static_cast<unsigned>(width)) << 8U;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.up.b32, var, delta, c, mask)
}
__CUDA_FP16_DECL__ __half2 __shfl_down_sync(const unsigned int mask, const __half2 var, const unsigned int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.down.b32, var, delta, c, mask)
}
__CUDA_FP16_DECL__ __half2 __shfl_xor_sync(const unsigned int mask, const __half2 var, const int laneMask, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.bfly.b32, var, laneMask, c, mask)
}

#undef __SHUFFLE_HALF2_MACRO
#undef __SHUFFLE_SYNC_HALF2_MACRO

#if defined(_NVHPC_CUDA) || !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 700)

__CUDA_FP16_DECL__ __half __shfl(const __half var, const int delta, const int width)
{
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl(temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_up(const __half var, const unsigned int delta, const int width)
{
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_up(temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_down(const __half var, const unsigned int delta, const int width)
{
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_down(temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_xor(const __half var, const int delta, const int width)
{
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_xor(temp1, delta, width);
    return __low2half(temp2);
}

#endif /* defined(_NVHPC_CUDA) || !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 700) */

__CUDA_FP16_DECL__ __half __shfl_sync(const unsigned int mask, const __half var, const int srcLane, const int width)
{
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_sync(mask, temp1, srcLane, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_up_sync(const unsigned int mask, const __half var, const unsigned int delta, const int width)
{
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_up_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_down_sync(const unsigned int mask, const __half var, const unsigned int delta, const int width)
{
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_down_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_xor_sync(const unsigned int mask, const __half var, const int laneMask, const int width)
{
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_xor_sync(mask, temp1, laneMask, width);
    return __low2half(temp2);
}

#endif /* !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300) || defined(_NVHPC_CUDA) */
/******************************************************************************
*               __half and __half2 __ldg,__ldcg,__ldca,__ldcs                *
******************************************************************************/

#if defined(__cplusplus) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 320) || defined(_NVHPC_CUDA))
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/
__CUDA_FP16_DECL__ __half2 __ldg(const  __half2 *const ptr)
{
    __half2 ret;
    asm ("ld.global.nc.b32 %0, [%1];"  : "=r"(__HALF2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half __ldg(const __half *const ptr)
{
    __half ret;
    asm ("ld.global.nc.b16 %0, [%1];"  : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half2 __ldcg(const  __half2 *const ptr)
{
    __half2 ret;
    asm ("ld.global.cg.b32 %0, [%1];"  : "=r"(__HALF2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half __ldcg(const __half *const ptr)
{
    __half ret;
    asm ("ld.global.cg.b16 %0, [%1];"  : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half2 __ldca(const  __half2 *const ptr)
{
    __half2 ret;
    asm ("ld.global.ca.b32 %0, [%1];"  : "=r"(__HALF2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half __ldca(const __half *const ptr)
{
    __half ret;
    asm ("ld.global.ca.b16 %0, [%1];"  : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half2 __ldcs(const  __half2 *const ptr)
{
    __half2 ret;
    asm ("ld.global.cs.b32 %0, [%1];"  : "=r"(__HALF2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half __ldcs(const __half *const ptr)
{
    __half ret;
    asm ("ld.global.cs.b16 %0, [%1];"  : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half2 __ldlu(const  __half2 *const ptr)
{
    __half2 ret;
    asm ("ld.global.lu.b32 %0, [%1];"  : "=r"(__HALF2_TO_UI(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}
__CUDA_FP16_DECL__ __half __ldlu(const __half *const ptr)
{
    __half ret;
    asm ("ld.global.lu.b16 %0, [%1];"  : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}
__CUDA_FP16_DECL__ __half2 __ldcv(const  __half2 *const ptr)
{
    __half2 ret;
    asm ("ld.global.cv.b32 %0, [%1];"  : "=r"(__HALF2_TO_UI(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}
__CUDA_FP16_DECL__ __half __ldcv(const __half *const ptr)
{
    __half ret;
    asm ("ld.global.cv.b16 %0, [%1];"  : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}
__CUDA_FP16_DECL__ void __stwb(__half2 *const ptr, const __half2 value)
{
    asm ("st.global.wb.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__HALF2_TO_CUI(value)) : "memory");
}
__CUDA_FP16_DECL__ void __stwb(__half *const ptr, const __half value)
{
    asm ("st.global.wb.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__HALF_TO_CUS(value)) : "memory");
}
__CUDA_FP16_DECL__ void __stcg(__half2 *const ptr, const __half2 value)
{
    asm ("st.global.cg.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__HALF2_TO_CUI(value)) : "memory");
}
__CUDA_FP16_DECL__ void __stcg(__half *const ptr, const __half value)
{
    asm ("st.global.cg.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__HALF_TO_CUS(value)) : "memory");
}
__CUDA_FP16_DECL__ void __stcs(__half2 *const ptr, const __half2 value)
{
    asm ("st.global.cs.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__HALF2_TO_CUI(value)) : "memory");
}
__CUDA_FP16_DECL__ void __stcs(__half *const ptr, const __half value)
{
    asm ("st.global.cs.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__HALF_TO_CUS(value)) : "memory");
}
__CUDA_FP16_DECL__ void __stwt(__half2 *const ptr, const __half2 value)
{
    asm ("st.global.wt.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__HALF2_TO_CUI(value)) : "memory");
}
__CUDA_FP16_DECL__ void __stwt(__half *const ptr, const __half value)
{
    asm ("st.global.wt.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__HALF_TO_CUS(value)) : "memory");
}
#undef __LDG_PTR
#endif /* defined(__cplusplus) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 320) || defined(_NVHPC_CUDA)) */
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

/******************************************************************************
*                             __half2 comparison                             *
******************************************************************************/
#define __COMPARISON_OP_HALF2_MACRO(name) /* do */ {\
   __half2 val; \
   asm( "{ " __CUDA_FP16_STRINGIFY(name) ".f16x2.f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); \
   return val; \
} /* while(0) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __heq2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.eq)
,
    __half2_raw val;
    val.x = __heq(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __heq(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hne2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.ne)
,
    __half2_raw val;
    val.x = __hne(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hne(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hle2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.le)
,
    __half2_raw val;
    val.x = __hle(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hle(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hge2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.ge)
,
    __half2_raw val;
    val.x = __hge(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hge(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hlt2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.lt)
,
    __half2_raw val;
    val.x = __hlt(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hlt(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hgt2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.gt)
,
    __half2_raw val;
    val.x = __hgt(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hgt(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hequ2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.equ)
,
    __half2_raw val;
    val.x = __hequ(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hequ(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hneu2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.neu)
,
    __half2_raw val;
    val.x = __hneu(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hneu(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hleu2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.leu)
,
    __half2_raw val;
    val.x = __hleu(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hleu(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hgeu2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.geu)
,
    __half2_raw val;
    val.x = __hgeu(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hgeu(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hltu2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.ltu)
,
    __half2_raw val;
    val.x = __hltu(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hltu(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hgtu2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO(set.gtu)
,
    __half2_raw val;
    val.x = __hgtu(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hgtu(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    return __half2(val);
)
}
#undef __COMPARISON_OP_HALF2_MACRO
/******************************************************************************
*                 __half2 comparison with mask output                        *
******************************************************************************/
#define __COMPARISON_OP_HALF2_MACRO_MASK(name) /* do */ {\
   unsigned val; \
   asm( "{ " __CUDA_FP16_STRINGIFY(name) ".u32.f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); \
   return val; \
} /* while(0) */
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __heq2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.eq)
,
    const unsigned short px = __heq(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __heq(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hne2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.ne)
,
    const unsigned short px = __hne(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hne(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hle2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.le)
,
    const unsigned short px = __hle(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hle(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hge2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.ge)
,
    const unsigned short px = __hge(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hge(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hlt2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.lt)
,
    const unsigned short px = __hlt(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hlt(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hgt2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.gt)
,
    const unsigned short px = __hgt(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hgt(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hequ2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.equ)
,
    const unsigned short px = __hequ(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hequ(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hneu2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.neu)
,
    const unsigned short px = __hneu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hneu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hleu2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.leu)
,
    const unsigned short px = __hleu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hleu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hgeu2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.geu)
,
    const unsigned short px = __hgeu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hgeu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hltu2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.ltu)
,
    const unsigned short px = __hltu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hltu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __hgtu2_mask(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF2_MACRO_MASK(set.gtu)
,
    const unsigned short px = __hgtu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hgtu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
#undef __COMPARISON_OP_HALF2_MACRO_MASK

__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbeq2(const __half2 a, const __half2 b)
{
    const unsigned mask = __heq2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbne2(const __half2 a, const __half2 b)
{
    const unsigned mask = __hne2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hble2(const __half2 a, const __half2 b)
{
    const unsigned mask = __hle2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbge2(const __half2 a, const __half2 b)
{
    const unsigned mask = __hge2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hblt2(const __half2 a, const __half2 b)
{
    const unsigned mask = __hlt2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbgt2(const __half2 a, const __half2 b)
{
    const unsigned mask = __hgt2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbequ2(const __half2 a, const __half2 b)
{
    const unsigned mask = __hequ2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbneu2(const __half2 a, const __half2 b)
{
    const unsigned mask = __hneu2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbleu2(const __half2 a, const __half2 b)
{
    const unsigned mask = __hleu2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbgeu2(const __half2 a, const __half2 b)
{
    const unsigned mask = __hgeu2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbltu2(const __half2 a, const __half2 b)
{
    const unsigned mask = __hltu2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hbgtu2(const __half2 a, const __half2 b)
{
    const unsigned mask = __hgtu2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
/******************************************************************************
*                             __half comparison                              *
******************************************************************************/
#define __COMPARISON_OP_HALF_MACRO(name) /* do */ {\
   unsigned short val; \
   asm( "{ .reg .pred __$temp3;\n" \
        "  setp." __CUDA_FP16_STRINGIFY(name) ".f16  __$temp3, %1, %2;\n" \
        "  selp.u16 %0, 1, 0, __$temp3;}" \
        : "=h"(val) : "h"(__HALF_TO_CUS(a)), "h"(__HALF_TO_CUS(b))); \
   return (val != 0U) ? true : false; \
} /* while(0) */
__CUDA_HOSTDEVICE_FP16_DECL__ bool __heq(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(eq)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa == fb);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hne(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(ne)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa != fb) && (!__hisnan(a)) && (!__hisnan(b));
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hle(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(le)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa <= fb);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hge(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(ge)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa >= fb);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hlt(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(lt)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa < fb);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hgt(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(gt)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa > fb);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hequ(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(equ)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa == fb) || (__hisnan(a)) || (__hisnan(b));
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hneu(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(neu)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa != fb);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hleu(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(leu)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa <= fb) || (__hisnan(a)) || (__hisnan(b));
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hgeu(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(geu)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa >= fb) || (__hisnan(a)) || (__hisnan(b));
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hltu(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(ltu)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa < fb) || (__hisnan(a)) || (__hisnan(b));
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hgtu(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __COMPARISON_OP_HALF_MACRO(gtu)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa > fb) || (__hisnan(a)) || (__hisnan(b));
)
}
#undef __COMPARISON_OP_HALF_MACRO
/******************************************************************************
*                            __half2 arithmetic                             *
******************************************************************************/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hadd2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF2_MACRO(add)
,
    __half2 val;
    val.x = __hadd(a.x, b.x);
    val.y = __hadd(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hsub2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF2_MACRO(sub)
,
    __half2 val;
    val.x = __hsub(a.x, b.x);
    val.y = __hsub(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmul2(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF2_MACRO(mul)
,
    __half2 val;
    val.x = __hmul(a.x, b.x);
    val.y = __hmul(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hadd2_sat(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF2_MACRO(add.sat)
,
    __half2 val;
    val.x = __hadd_sat(a.x, b.x);
    val.y = __hadd_sat(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hsub2_sat(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF2_MACRO(sub.sat)
,
    __half2 val;
    val.x = __hsub_sat(a.x, b.x);
    val.y = __hsub_sat(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmul2_sat(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF2_MACRO(mul.sat)
,
    __half2 val;
    val.x = __hmul_sat(a.x, b.x);
    val.y = __hmul_sat(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hadd2_rn(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF2_MACRO(add.rn)
,
    __half2 val;
    val.x = __hadd_rn(a.x, b.x);
    val.y = __hadd_rn(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hsub2_rn(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF2_MACRO(sub.rn)
,
    __half2 val;
    val.x = __hsub_rn(a.x, b.x);
    val.y = __hsub_rn(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmul2_rn(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF2_MACRO(mul.rn)
,
    __half2 val;
    val.x = __hmul_rn(a.x, b.x);
    val.y = __hmul_rn(a.y, b.y);
    return val;
)
}
#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c)
{
    __TERNARY_OP_HALF2_MACRO(fma.rn)
}
__CUDA_FP16_DECL__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c)
{
    __TERNARY_OP_HALF2_MACRO(fma.rn.sat)
}
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __h2div(const __half2 a, const __half2 b) {
    __half ha = __low2half(a);
    __half hb = __low2half(b);

    const __half v1 = __hdiv(ha, hb);

    ha = __high2half(a);
    hb = __high2half(b);

    const __half v2 = __hdiv(ha, hb);

    return __halves2half2(v1, v2);
}

/******************************************************************************
*                             __half arithmetic                             *
******************************************************************************/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hadd(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF_MACRO(add)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return __float2half(fa + fb);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hsub(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF_MACRO(sub)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return __float2half(fa - fb);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmul(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF_MACRO(mul)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return __float2half(fa * fb);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hadd_sat(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF_MACRO(add.sat)
,
    return __hmin(__hmax(__hadd(a, b), CUDART_ZERO_FP16), CUDART_ONE_FP16);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hsub_sat(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF_MACRO(sub.sat)
,
    return __hmin(__hmax(__hsub(a, b), CUDART_ZERO_FP16), CUDART_ONE_FP16);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmul_sat(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF_MACRO(mul.sat)
,
    return __hmin(__hmax(__hmul(a, b), CUDART_ZERO_FP16), CUDART_ONE_FP16);
)
}

__CUDA_HOSTDEVICE_FP16_DECL__ __half __hadd_rn(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF_MACRO(add.rn)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return __float2half(fa + fb);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hsub_rn(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF_MACRO(sub.rn)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return __float2half(fa - fb);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmul_rn(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __BINARY_OP_HALF_MACRO(mul.rn)
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return __float2half(fa * fb);
)
}
#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c)
{
    __TERNARY_OP_HALF_MACRO(fma.rn)
}
__CUDA_FP16_DECL__ __half __hfma_sat(const __half a, const __half b, const __half c)
{
    __TERNARY_OP_HALF_MACRO(fma.rn.sat)
}
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hdiv(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __half v;
    __half abs;
    __half den;
    __HALF_TO_US(den) = 0x008FU;

    float rcp;
    const float fa = __half2float(a);
    const float fb = __half2float(b);

    asm("{rcp.approx.ftz.f32 %0, %1;\n}" :"=f"(rcp) : "f"(fb));

    float fv = rcp * fa;

    v = __float2half(fv);
    abs = __habs(v);
    if (__hlt(abs, den) && __hlt(__float2half(0.0f), abs))  {
        const float err = __fmaf_rn(-fb, fv, fa);
        fv = __fmaf_rn(rcp, err, fv);
        v = __float2half(fv);
    }
    return v;
,
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return __float2half(fa / fb);
)
}

/******************************************************************************
*                             __half2 functions                  *
******************************************************************************/
#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
#define __APPROX_FCAST(fun) /* do */ {\
   __half val;\
   asm("{.reg.b32         f;        \n"\
                " .reg.b16         r;        \n"\
                "  mov.b16         r,%1;     \n"\
                "  cvt.f32.f16     f,r;      \n"\
                "  " __CUDA_FP16_STRINGIFY(fun) ".approx.ftz.f32   f,f;  \n"\
                "  cvt.rn.f16.f32      r,f;  \n"\
                "  mov.b16         %0,r;     \n"\
                "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));\
   return val;\
} /* while(0) */
#define __APPROX_FCAST2(fun) /* do */ {\
   __half2 val;\
   asm("{.reg.b16         hl, hu;         \n"\
                " .reg.b32         fl, fu;         \n"\
                "  mov.b32         {hl, hu}, %1;   \n"\
                "  cvt.f32.f16     fl, hl;         \n"\
                "  cvt.f32.f16     fu, hu;         \n"\
                "  " __CUDA_FP16_STRINGIFY(fun) ".approx.ftz.f32   fl, fl;     \n"\
                "  " __CUDA_FP16_STRINGIFY(fun) ".approx.ftz.f32   fu, fu;     \n"\
                "  cvt.rn.f16.f32      hl, fl;     \n"\
                "  cvt.rn.f16.f32      hu, fu;     \n"\
                "  mov.b32         %0, {hl, hu};   \n"\
                "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));       \
   return val;\
} /* while(0) */
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530) || defined(_NVHPC_CUDA)
#define __SPEC_CASE2(i,r, spc, ulp) \
   "{.reg.b32 spc, ulp, p;\n"\
   "  mov.b32 spc," __CUDA_FP16_STRINGIFY(spc) ";\n"\
   "  mov.b32 ulp," __CUDA_FP16_STRINGIFY(ulp) ";\n"\
   "  set.eq.f16x2.f16x2 p," __CUDA_FP16_STRINGIFY(i) ", spc;\n"\
   "  fma.rn.f16x2 " __CUDA_FP16_STRINGIFY(r) ",p,ulp," __CUDA_FP16_STRINGIFY(r) ";\n}\n"
#define __SPEC_CASE(i,r, spc, ulp) \
   "{.reg.b16 spc, ulp, p;\n"\
   "  mov.b16 spc," __CUDA_FP16_STRINGIFY(spc) ";\n"\
   "  mov.b16 ulp," __CUDA_FP16_STRINGIFY(ulp) ";\n"\
   "  set.eq.f16.f16 p," __CUDA_FP16_STRINGIFY(i) ", spc;\n"\
   "  fma.rn.f16 " __CUDA_FP16_STRINGIFY(r) ",p,ulp," __CUDA_FP16_STRINGIFY(r) ";\n}\n"
static __device__ __forceinline__ float __float_simpl_sinf(float a);
static __device__ __forceinline__ float __float_simpl_cosf(float a);
__CUDA_FP16_DECL__ __half hsin(const __half a) {
    const float sl = __float_simpl_sinf(__half2float(a));
    __half r = __float2half_rn(sl);
    asm("{\n\t"
        "  .reg.b16 i,r,t;     \n\t"
        "  mov.b16 r, %0;      \n\t"
        "  mov.b16 i, %1;      \n\t"
        "  and.b16 t, r, 0x8000U; \n\t"
        "  abs.f16 r, r;   \n\t"
        "  abs.f16 i, i;   \n\t"
        __SPEC_CASE(i, r, 0X32B3U, 0x0800U)
        __SPEC_CASE(i, r, 0X5CB0U, 0x9000U)
        "  or.b16  r,r,t;      \n\t"
        "  mov.b16 %0, r;      \n"
        "}\n" : "+h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(a)));
    return r;
}
__CUDA_FP16_DECL__ __half2 h2sin(const __half2 a) {
    const float sl = __float_simpl_sinf(__half2float(a.x));
    const float sh = __float_simpl_sinf(__half2float(a.y));
    __half2 r = __floats2half2_rn(sl, sh);
    asm("{\n\t"
        "  .reg.b32 i,r,t;             \n\t"
        "  mov.b32 r, %0;              \n\t"
        "  mov.b32 i, %1;              \n\t"
        "  and.b32 t, r, 0x80008000U;   \n\t"
        "  abs.f16x2 r, r;   \n\t"
        "  abs.f16x2 i, i;   \n\t"
        __SPEC_CASE2(i, r, 0X32B332B3U, 0x08000800U)
        __SPEC_CASE2(i, r, 0X5CB05CB0U, 0x90009000U)
        "  or.b32  r, r, t;            \n\t"
        "  mov.b32 %0, r;              \n"
        "}\n" : "+r"(__HALF2_TO_UI(r)) : "r"(__HALF2_TO_CUI(a)));
    return r;
}
__CUDA_FP16_DECL__ __half hcos(const __half a) {
    const float cl = __float_simpl_cosf(__half2float(a));
    __half r = __float2half_rn(cl);
    asm("{\n\t"
        "  .reg.b16 i,r;        \n\t"
        "  mov.b16 r, %0;       \n\t"
        "  mov.b16 i, %1;       \n\t"
        "  abs.f16 i, i;        \n\t"
        __SPEC_CASE(i, r, 0X2B7CU, 0x1000U)
        "  mov.b16 %0, r;       \n"
        "}\n" : "+h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(a)));
    return r;
}
__CUDA_FP16_DECL__ __half2 h2cos(const __half2 a) {
    const float cl = __float_simpl_cosf(__half2float(a.x));
    const float ch = __float_simpl_cosf(__half2float(a.y));
    __half2 r = __floats2half2_rn(cl, ch);
    asm("{\n\t"
        "  .reg.b32 i,r;   \n\t"
        "  mov.b32 r, %0;  \n\t"
        "  mov.b32 i, %1;  \n\t"
        "  abs.f16x2 i, i; \n\t"
        __SPEC_CASE2(i, r, 0X2B7C2B7CU, 0x10001000U)
        "  mov.b32 %0, r;  \n"
        "}\n" : "+r"(__HALF2_TO_UI(r)) : "r"(__HALF2_TO_CUI(a)));
    return r;
}
static __device__ __forceinline__ float __internal_trig_reduction_kernel(const float a, unsigned int *const quadrant)
{
    const float ar = __fmaf_rn(a, 0.636619772F, 12582912.0F);
    const unsigned q = __float_as_uint(ar);
    const float j = __fsub_rn(ar, 12582912.0F);
    float t = __fmaf_rn(j, -1.5707962512969971e+000F, a);
    t = __fmaf_rn(j, -7.5497894158615964e-008F, t);
    *quadrant = q;
    return t;
}
static __device__ __forceinline__ float __internal_sin_cos_kernel(const float x, const unsigned int i)
{
    float z;
    const float x2 = x*x;
    float a8;
    float a6;
    float a4;
    float a2;
    float a1;
    float a0;

    if ((i & 1U) != 0U) {
        // cos
        a8 =  2.44331571e-5F;
        a6 = -1.38873163e-3F;
        a4 =  4.16666457e-2F;
        a2 = -5.00000000e-1F;
        a1 = x2;
        a0 = 1.0F;
    }
    else {
        // sin
        a8 = -1.95152959e-4F;
        a6 =  8.33216087e-3F;
        a4 = -1.66666546e-1F;
        a2 = 0.0F;
        a1 = x;
        a0 = x;
    }

    z = __fmaf_rn(a8, x2, a6);
    z = __fmaf_rn(z, x2, a4);
    z = __fmaf_rn(z, x2, a2);
    z = __fmaf_rn(z, a1, a0);

    if ((i & 2U) != 0U) {
        z = -z;
    }
    return z;
}
static __device__ __forceinline__ float __float_simpl_sinf(float a)
{
    float z;
    unsigned i;
    a = __internal_trig_reduction_kernel(a, &i);
    z = __internal_sin_cos_kernel(a, i);
    return z;
}
static __device__ __forceinline__ float __float_simpl_cosf(float a)
{
    float z;
    unsigned i;
    a = __internal_trig_reduction_kernel(a, &i);
    z = __internal_sin_cos_kernel(a, (i & 0x3U) + 1U);
    return z;
}

__CUDA_FP16_DECL__ __half hexp(const __half a) {
    __half val;
    asm("{.reg.b32         f, C, nZ;       \n"
        " .reg.b16         h,r;            \n"
        "  mov.b16         h,%1;           \n"
        "  cvt.f32.f16     f,h;            \n"
        "  mov.b32         C, 0x3fb8aa3bU; \n"
        "  mov.b32         nZ, 0x80000000U;\n"
        "  fma.rn.f32      f,f,C,nZ;       \n"
        "  ex2.approx.ftz.f32  f,f;        \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        __SPEC_CASE(h, r, 0X1F79U, 0x9400U)
        __SPEC_CASE(h, r, 0X25CFU, 0x9400U)
        __SPEC_CASE(h, r, 0XC13BU, 0x0400U)
        __SPEC_CASE(h, r, 0XC1EFU, 0x0200U)
        "  mov.b16         %0,r;           \n"
        "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2exp(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         h,r,fl,fu,C,nZ; \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  mov.b32         h, %1;          \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  mov.b32         C, 0x3fb8aa3bU; \n"
        "  mov.b32         nZ, 0x80000000U;\n"
        "  fma.rn.f32      fl,fl,C,nZ;     \n"
        "  fma.rn.f32      fu,fu,C,nZ;     \n"
        "  ex2.approx.ftz.f32  fl, fl;     \n"
        "  ex2.approx.ftz.f32  fu, fu;     \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        __SPEC_CASE2(h, r, 0X1F791F79U, 0x94009400U)
        __SPEC_CASE2(h, r, 0X25CF25CFU, 0x94009400U)
        __SPEC_CASE2(h, r, 0XC13BC13BU, 0x04000400U)
        __SPEC_CASE2(h, r, 0XC1EFC1EFU, 0x02000200U)
        "  mov.b32         %0, r;  \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
#endif /* !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530) || defined(_NVHPC_CUDA) */
__CUDA_FP16_DECL__ __half hexp2(const __half a) {
    __half val;
    asm("{.reg.b32         f, ULP;         \n"
        " .reg.b16         r;              \n"
        "  mov.b16         r,%1;           \n"
        "  cvt.f32.f16     f,r;            \n"
        "  ex2.approx.ftz.f32      f,f;    \n"
        "  mov.b32         ULP, 0x33800000U;\n"
        "  fma.rn.f32      f,f,ULP,f;      \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        "  mov.b16         %0,r;           \n"
        "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2exp2(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         fl, fu, ULP;    \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  ex2.approx.ftz.f32  fl, fl;     \n"
        "  ex2.approx.ftz.f32  fu, fu;     \n"
        "  mov.b32         ULP, 0x33800000U;\n"
        "  fma.rn.f32      fl,fl,ULP,fl;   \n"
        "  fma.rn.f32      fu,fu,ULP,fu;   \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         %0, {hl, hu};   \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ __half hexp10(const __half a) {
    __half val;
    asm("{.reg.b16         h,r;            \n"
        " .reg.b32         f, C, nZ;       \n"
        "  mov.b16         h, %1;          \n"
        "  cvt.f32.f16     f, h;           \n"
        "  mov.b32         C, 0x40549A78U; \n"
        "  mov.b32         nZ, 0x80000000U;\n"
        "  fma.rn.f32      f,f,C,nZ;       \n"
        "  ex2.approx.ftz.f32  f, f;       \n"
        "  cvt.rn.f16.f32      r, f;       \n"
        __SPEC_CASE(h, r, 0x34DEU, 0x9800U)
        __SPEC_CASE(h, r, 0x9766U, 0x9000U)
        __SPEC_CASE(h, r, 0x9972U, 0x1000U)
        __SPEC_CASE(h, r, 0xA5C4U, 0x1000U)
        __SPEC_CASE(h, r, 0xBF0AU, 0x8100U)
        "  mov.b16         %0, r;          \n"
        "}":"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2exp10(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         h,r,fl,fu,C,nZ; \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  mov.b32         h, %1;          \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  mov.b32         C, 0x40549A78U; \n"
        "  mov.b32         nZ, 0x80000000U;\n"
        "  fma.rn.f32      fl,fl,C,nZ;     \n"
        "  fma.rn.f32      fu,fu,C,nZ;     \n"
        "  ex2.approx.ftz.f32  fl, fl;     \n"
        "  ex2.approx.ftz.f32  fu, fu;     \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        __SPEC_CASE2(h, r, 0x34DE34DEU, 0x98009800U)
        __SPEC_CASE2(h, r, 0x97669766U, 0x90009000U)
        __SPEC_CASE2(h, r, 0x99729972U, 0x10001000U)
        __SPEC_CASE2(h, r, 0xA5C4A5C4U, 0x10001000U)
        __SPEC_CASE2(h, r, 0xBF0ABF0AU, 0x81008100U)
        "  mov.b32         %0, r;  \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ __half hlog2(const __half a) {
    __half val;
    asm("{.reg.b16         h, r;           \n"
        " .reg.b32         f;              \n"
        "  mov.b16         h, %1;          \n"
        "  cvt.f32.f16     f, h;           \n"
        "  lg2.approx.ftz.f32  f, f;       \n"
        "  cvt.rn.f16.f32      r, f;       \n"
        __SPEC_CASE(r, r, 0xA2E2U, 0x8080U)
        __SPEC_CASE(r, r, 0xBF46U, 0x9400U)
        "  mov.b16         %0, r;          \n"
        "}":"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2log2(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         fl, fu, r, p;   \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  lg2.approx.ftz.f32  fl, fl;     \n"
        "  lg2.approx.ftz.f32  fu, fu;     \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        __SPEC_CASE2(r, r, 0xA2E2A2E2U, 0x80808080U)
        __SPEC_CASE2(r, r, 0xBF46BF46U, 0x94009400U)
        "  mov.b32         %0, r;          \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ __half hlog(const __half a) {
    __half val;
    asm("{.reg.b32         f, C;           \n"
        " .reg.b16         r,h;            \n"
        "  mov.b16         h,%1;           \n"
        "  cvt.f32.f16     f,h;            \n"
        "  lg2.approx.ftz.f32  f,f;        \n"
        "  mov.b32         C, 0x3f317218U;  \n"
        "  mul.f32         f,f,C;          \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        __SPEC_CASE(h, r, 0X160DU, 0x9C00U)
        __SPEC_CASE(h, r, 0X3BFEU, 0x8010U)
        __SPEC_CASE(h, r, 0X3C0BU, 0x8080U)
        __SPEC_CASE(h, r, 0X6051U, 0x1C00U)
        "  mov.b16         %0,r;           \n"
        "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2log(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;             \n"
        " .reg.b32         r, fl, fu, C, h;    \n"
        "  mov.b32         {hl, hu}, %1;       \n"
        "  mov.b32         h, %1;              \n"
        "  cvt.f32.f16     fl, hl;             \n"
        "  cvt.f32.f16     fu, hu;             \n"
        "  lg2.approx.ftz.f32  fl, fl;         \n"
        "  lg2.approx.ftz.f32  fu, fu;         \n"
        "  mov.b32         C, 0x3f317218U;     \n"
        "  mul.f32         fl,fl,C;            \n"
        "  mul.f32         fu,fu,C;            \n"
        "  cvt.rn.f16.f32      hl, fl;         \n"
        "  cvt.rn.f16.f32      hu, fu;         \n"
        "  mov.b32         r, {hl, hu};        \n"
        __SPEC_CASE2(h, r, 0X160D160DU, 0x9C009C00U)
        __SPEC_CASE2(h, r, 0X3BFE3BFEU, 0x80108010U)
        __SPEC_CASE2(h, r, 0X3C0B3C0BU, 0x80808080U)
        __SPEC_CASE2(h, r, 0X60516051U, 0x1C001C00U)
        "  mov.b32         %0, r;              \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ __half hlog10(const __half a) {
    __half val;
    asm("{.reg.b16         h, r;           \n"
        " .reg.b32         f, C;           \n"
        "  mov.b16         h, %1;          \n"
        "  cvt.f32.f16     f, h;           \n"
        "  lg2.approx.ftz.f32  f, f;       \n"
        "  mov.b32         C, 0x3E9A209BU; \n"
        "  mul.f32         f,f,C;          \n"
        "  cvt.rn.f16.f32      r, f;       \n"
        __SPEC_CASE(h, r, 0x338FU, 0x1000U)
        __SPEC_CASE(h, r, 0x33F8U, 0x9000U)
        __SPEC_CASE(h, r, 0x57E1U, 0x9800U)
        __SPEC_CASE(h, r, 0x719DU, 0x9C00U)
        "  mov.b16         %0, r;          \n"
        "}":"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2log10(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;             \n"
        " .reg.b32         r, fl, fu, C, h;    \n"
        "  mov.b32         {hl, hu}, %1;       \n"
        "  mov.b32         h, %1;              \n"
        "  cvt.f32.f16     fl, hl;             \n"
        "  cvt.f32.f16     fu, hu;             \n"
        "  lg2.approx.ftz.f32  fl, fl;         \n"
        "  lg2.approx.ftz.f32  fu, fu;         \n"
        "  mov.b32         C, 0x3E9A209BU;     \n"
        "  mul.f32         fl,fl,C;            \n"
        "  mul.f32         fu,fu,C;            \n"
        "  cvt.rn.f16.f32      hl, fl;         \n"
        "  cvt.rn.f16.f32      hu, fu;         \n"
        "  mov.b32         r, {hl, hu};        \n"
        __SPEC_CASE2(h, r, 0x338F338FU, 0x10001000U)
        __SPEC_CASE2(h, r, 0x33F833F8U, 0x90009000U)
        __SPEC_CASE2(h, r, 0x57E157E1U, 0x98009800U)
        __SPEC_CASE2(h, r, 0x719D719DU, 0x9C009C00U)
        "  mov.b32         %0, r;              \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
#undef __SPEC_CASE2
#undef __SPEC_CASE
#endif /* !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530) || defined(_NVHPC_CUDA) */
__CUDA_FP16_DECL__ __half2 h2rcp(const __half2 a) {
    __APPROX_FCAST2(rcp)
}
__CUDA_FP16_DECL__ __half hrcp(const __half a) {
    __APPROX_FCAST(rcp)
}
__CUDA_FP16_DECL__ __half2 h2rsqrt(const __half2 a) {
    __APPROX_FCAST2(rsqrt)
}
__CUDA_FP16_DECL__ __half hrsqrt(const __half a) {
    __APPROX_FCAST(rsqrt)
}
__CUDA_FP16_DECL__ __half2 h2sqrt(const __half2 a) {
    __APPROX_FCAST2(sqrt)
}
__CUDA_FP16_DECL__ __half hsqrt(const __half a) {
    __APPROX_FCAST(sqrt)
}
#undef __APPROX_FCAST
#undef __APPROX_FCAST2
#endif /* defined(_NVHPC_CUDA) || defined(__CUDACC__) */
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hisnan2(const __half2 a)
{
    __half2 r;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    asm("{set.nan.f16x2.f16x2 %0,%1,%2;\n}"
        :"=r"(__HALF2_TO_UI(r)) : "r"(__HALF2_TO_CUI(a)), "r"(__HALF2_TO_CUI(a)));
,
    __half2_raw val;
    val.x = __hisnan(a.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
    val.y = __hisnan(a.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
    r = __half2(val);
)
    return r;
}
__CUDA_HOSTDEVICE_FP16_DECL__ bool __hisnan(const __half a)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __half r;
    asm("{set.nan.f16.f16 %0,%1,%2;\n}"
        :"=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(a)), "h"(__HALF_TO_CUS(a)));
    return __HALF_TO_CUS(r) != 0U;
,
    const __half_raw hr = static_cast<__half_raw>(a);
    return ((hr.x & (unsigned short)0x7FFFU) > (unsigned short)0x7C00U);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hneg2(const __half2 a)
{
    __half2 r;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    asm("{neg.f16x2 %0,%1;\n}"
        :"=r"(__HALF2_TO_UI(r)) : "r"(__HALF2_TO_CUI(a)));
,
    r.x = __hneg(a.x);
    r.y = __hneg(a.y);
)
    return r;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hneg(const __half a)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __half r;
    asm("{neg.f16 %0,%1;\n}"
        :"=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(a)));
    return r;
,
    const float fa = __half2float(a);
    return __float2half(-fa);
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __habs2(const __half2 a)
{
    __half2 r;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    asm("{abs.f16x2 %0,%1;\n}"
        :"=r"(__HALF2_TO_UI(r)) : "r"(__HALF2_TO_CUI(a)));
,
    r.x = __habs(a.x);
    r.y = __habs(a.y);
)
    return r;
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __habs(const __half a)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
    __half r;
    asm("{abs.f16 %0,%1;\n}"
        :"=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(a)));
    return r;
,
    __half_raw abs_a_raw = static_cast<__half_raw>(a);
    abs_a_raw.x &= (unsigned short)0x7FFFU;
    if (abs_a_raw.x > (unsigned short)0x7C00U)
    {
        // return canonical NaN
        abs_a_raw.x = (unsigned short)0x7FFFU;
    }
    return static_cast<__half>(abs_a_raw);
)
}
#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ __half2 __hcmadd(const __half2 a, const __half2 b, const __half2 c)
{
    // fast version of complex multiply-accumulate
    // (a.re, a.im) * (b.re, b.im) + (c.re, c.im)
    // acc.re = (c.re + a.re*b.re) - a.im*b.im
    // acc.im = (c.im + a.re*b.im) + a.im*b.re
    __half real_tmp =  __hfma(a.x, b.x, c.x);
    __half img_tmp  =  __hfma(a.x, b.y, c.y);
    real_tmp = __hfma(__hneg(a.y), b.y, real_tmp);
    img_tmp  = __hfma(a.y,         b.x, img_tmp);
    return make_half2(real_tmp, img_tmp);
}
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */

__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmax_nan(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __BINARY_OP_HALF_MACRO(max.NaN)
,
    __half maxval;
    if (__hisnan(a) || __hisnan(b))
    {
        maxval = CUDART_NAN_FP16;
    }
    else
    {
        maxval = __hmax(a, b);
    }
    return maxval;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half __hmin_nan(const __half a, const __half b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __BINARY_OP_HALF_MACRO(min.NaN)
,
    __half minval;
    if (__hisnan(a) || __hisnan(b))
    {
        minval = CUDART_NAN_FP16;
    }
    else
    {
        minval = __hmin(a, b);
    }
    return minval;
)
}

#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ __half __hfma_relu(const __half a, const __half b, const __half c)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __TERNARY_OP_HALF_MACRO(fma.rn.relu)
,
    return __hmax_nan(__hfma(a, b, c), CUDART_ZERO_FP16);
)
}
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */

__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmax2_nan(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __BINARY_OP_HALF2_MACRO(max.NaN)
,
    __half2 result = __hmax2(a, b);
    if (__hisnan(a.x) || __hisnan(b.x))
    {
        result.x = CUDART_NAN_FP16;
    }
    if (__hisnan(a.y) || __hisnan(b.y))
    {
        result.y = CUDART_NAN_FP16;
    }
    return result;
)
}
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __hmin2_nan(const __half2 a, const __half2 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __BINARY_OP_HALF2_MACRO(min.NaN)
,
    __half2 result = __hmin2(a, b);
    if (__hisnan(a.x) || __hisnan(b.x))
    {
        result.x = CUDART_NAN_FP16;
    }
    if (__hisnan(a.y) || __hisnan(b.y))
    {
        result.y = CUDART_NAN_FP16;
    }
    return result;
)
}
#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__ __half2 __hfma2_relu(const __half2 a, const __half2 b, const __half2 c)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __TERNARY_OP_HALF2_MACRO(fma.rn.relu)
,
    __half2_raw hzero;
    hzero.x = (unsigned short)0U;
    hzero.y = (unsigned short)0U;
    return __hmax2_nan(__hfma2(a, b, c), __half2(hzero));
)
}
#endif /* defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)) || defined(_NVHPC_CUDA) */

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/* Define __PTR for atomicAdd prototypes below, undef after done */
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __PTR   "l"
#else
#define __PTR   "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/

__CUDA_FP16_DECL__  __half2 atomicAdd(__half2 *const address, const __half2 val) {
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
    __half2 r;
    asm volatile ("{ atom.add.noftz.f16x2 %0,[%1],%2; }\n"
                  : "=r"(__HALF2_TO_UI(r)) : __PTR(address), "r"(__HALF2_TO_CUI(val))
                  : "memory");
    return r;
,
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        __half2 new_val = __hadd2(val, *(__half2*)&assumed);
        old = atomicCAS(address_as_uint, assumed, *(unsigned int*)&new_val);
    } while (assumed != old);
    return *(__half2*)&old;
)
}

#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA)
__CUDA_FP16_DECL__  __half atomicAdd(__half *const address, const __half val) {
    __half r;
    asm volatile ("{ atom.add.noftz.f16 %0,[%1],%2; }\n"
                  : "=h"(__HALF_TO_US(r))
                  : __PTR(address), "h"(__HALF_TO_CUS(val))
                  : "memory");
    return r;
}
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA) */

#undef __PTR
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
#endif /* !(defined __DOXYGEN_ONLY__) */
#endif /* defined(__cplusplus) */

#undef __TERNARY_OP_HALF2_MACRO
#undef __TERNARY_OP_HALF_MACRO
#undef __BINARY_OP_HALF2_MACRO
#undef __BINARY_OP_HALF_MACRO

#undef __CUDA_HOSTDEVICE_FP16_DECL__
#undef __CUDA_FP16_DECL__

#undef __HALF_TO_US
#undef __HALF_TO_CUS
#undef __HALF2_TO_UI
#undef __HALF2_TO_CUI
#undef __CUDA_FP16_CONSTEXPR__

#if defined(__CPP_VERSION_AT_LEAST_11_FP16)
#undef __CPP_VERSION_AT_LEAST_11_FP16
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */

#undef ___CUDA_FP16_STRINGIFY_INNERMOST
#undef __CUDA_FP16_STRINGIFY

#endif /* end of include guard: __CUDA_FP16_HPP__ */
