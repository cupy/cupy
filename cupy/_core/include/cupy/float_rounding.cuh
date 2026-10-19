#pragma once

// Round to odd so the following RNE float -> bfloat16 conversion is exact
// (Boldo & Melquiond 2005).  Bit-level: -ftz=true flushes float subnormals.
inline __device__ float double_to_float_round_to_odd(double v) {
  union {
    double d;
    unsigned long long u;
  } pun;
  pun.d = v;
  const unsigned long long bits = pun.u;

  const unsigned int sign = (bits >> 32) & 0x80000000u;
  const unsigned int exp = (bits >> 52) & 0x7ffu;
  const unsigned long long mant = bits & 0xfffffffffffffull;

  if (exp == 0x7ffu) {
    return __uint_as_float(
        sign | (mant == 0 ? 0x7f800000u : 0x7fc00000u));
  }
  if (exp == 0u) {
    return __uint_as_float(sign | (mant != 0));
  }

  const int e = static_cast<int>(exp) - 1023;
  if (e > 127) {
    return __uint_as_float(sign | 0x7f7fffffu);
  }
  if (e < -149) {
    return __uint_as_float(sign | 1u);
  }
  if (e >= -126) {
    unsigned int top = static_cast<unsigned int>(mant >> 29);
    if ((mant & ((1ull << 29) - 1)) != 0) {
      top |= 1u;
    }
    return __uint_as_float(
        sign | (static_cast<unsigned int>(e + 127) << 23) | top);
  }
  // value = (1.mant) * 2**e in units of 2**-149.
  const int shift = -e - 97;
  const unsigned long long sig = (1ull << 52) | mant;
  unsigned int m = static_cast<unsigned int>(sig >> shift);
  if ((sig & ((1ull << shift) - 1)) != 0) {
    m |= 1u;
  }
  return __uint_as_float(sign | m);
}


// Use native 32-bit conversions, then jam if any integer bits were lost.
// Rounding toward zero keeps the float in range for the cast back.
inline __device__ float integer_to_float_round_to_odd(int v) {
  const float truncated = __int2float_rz(v);
  const bool inexact = static_cast<int>(truncated) != v;
  return __uint_as_float(__float_as_uint(truncated) | inexact);
}


inline __device__ float integer_to_float_round_to_odd(unsigned int v) {
  const float truncated = __uint2float_rz(v);
  const bool inexact = static_cast<unsigned int>(truncated) != v;
  return __uint_as_float(__float_as_uint(truncated) | inexact);
}


// Keep the top 24 bits and set the LSB when a dropped bit is nonzero.
inline __device__ float integer_to_float_round_to_odd(unsigned long long v) {
  if (v == 0) {
    return 0.0f;
  }
  const int bit_count = 64 - __clzll(v);
  const int shift = bit_count > 24 ? bit_count - 24 : 0;
  const unsigned long long kept = v >> shift;
  unsigned long long rounded = kept;
  if (shift > 0 && (v & ((1ull << shift) - 1)) != 0) {
    rounded |= 1u;
  }
  return ldexpf(static_cast<float>(rounded), shift);
}


inline __device__ float integer_to_float_round_to_odd(long long v) {
  const bool negative = v < 0;
  // Magnitude, also correct for the minimum value.
  const unsigned long long magnitude =
      negative ? 0ull - static_cast<unsigned long long>(v)
               : static_cast<unsigned long long>(v);
  const float result = integer_to_float_round_to_odd(magnitude);
  return negative ? -result : result;
}
