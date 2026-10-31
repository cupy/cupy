#pragma once

inline __device__ unsigned short double_to_bfloat16_round_to_neven(double v) {
  union {
    double d;
    unsigned long long u;
  } pun;
  pun.d = v;
  const unsigned long long bits = pun.u;
  const unsigned short sign = static_cast<unsigned short>(bits >> 48) & 0x8000u;
  const unsigned int exp = static_cast<unsigned int>((bits >> 52) & 0x7ffu);
  const unsigned long long mant = bits & 0xfffffffffffffull;

  if (exp == 0x7ffu) {  // Infinity, or NaN made quiet.
    return static_cast<unsigned short>(
        sign | 0x7f80u | (mant != 0 ? 0x40u : 0u));
  }
  const int e = static_cast<int>(exp) - 1023;
  if (e > 127) {  // Past the overflow threshold: rounds to infinity.
    return static_cast<unsigned short>(sign | 0x7f80u);
  }
  if (e < -134) {  // Below half of the smallest bfloat16 subnormal.
    return sign;
  }
  if (e >= -126) {
    // Normal range.  The absorbing add performs the rounding and lets the
    // carry roll into the exponent, topping out at infinity.
    const unsigned long long bias =
        (1ull << 44) - 1 + ((mant >> 45) & 1ull);
    return static_cast<unsigned short>(
        sign | ((e + 127) * 128 + static_cast<int>((mant + bias) >> 45)));
  }
  // Subnormal bfloat16: value = sig * 2^(e - 52) on a 2^-133 grid.
  const unsigned long long sig = (1ull << 52) | mant;
  const int shift = -e - 81;
  const unsigned long long bias =
      (1ull << (shift - 1)) - 1 + ((sig >> shift) & 1ull);
  return static_cast<unsigned short>(sign | ((sig + bias) >> shift));
}

inline __device__ unsigned short integer_to_bfloat16_round_to_neven(
    bool negative, unsigned long long magnitude) {
  if (magnitude == 0) {
    return static_cast<unsigned short>(negative) << 15;
  }
  const int lz = __clzll(magnitude);
  // Normalized fraction without the implicit leading one.
  const unsigned long long fraction =
      (magnitude << lz) & 0x7fffffffffffffffull;
  // Half of the dropped bits minus one, plus the kept LSB: the absorbing add
  // rounds and the carry rolls into the exponent.
  const unsigned long long bias =
      0x007fffffffffffffull + ((fraction >> 56) & 1ull);
  return static_cast<unsigned short>(
      (static_cast<unsigned short>(negative) << 15) +
      ((190 - lz) * 128 + static_cast<int>((fraction + bias) >> 56)));
}

inline __device__ unsigned short integer_to_bfloat16_round_to_neven(
    long long v) {
  const bool negative = v < 0;
  // Magnitude, also correct for the minimum value.
  const unsigned long long magnitude =
      negative ? 0ull - static_cast<unsigned long long>(v)
               : static_cast<unsigned long long>(v);
  return integer_to_bfloat16_round_to_neven(negative, magnitude);
}

inline __device__ unsigned short integer_to_bfloat16_round_to_neven(
    unsigned long long v) {
  return integer_to_bfloat16_round_to_neven(false, v);
}
