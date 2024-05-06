#pragma once

#include <cuda/std/type_traits>


template<typename T, int maxlen_>
class NumPyString {
public:
    static const int maxlen = maxlen_;
    // TODO: 0 is possible, but C doesn't like it here (for good reasons)
    //       there may be a way to tell C++ that empty is OK/possible?
    T data[maxlen_ ? maxlen_ : 1];

    __host__ __device__ int strlen() {
        int len = maxlen;
        while (len > 0 && this->data[len-1] == 0) {
            len--;
        }
        return len;
    }

    __host__ __device__ NumPyString () {}

    template<typename OT, int Olen>
    __host__ __device__ NumPyString (const NumPyString<OT, Olen> &other)
    {
        // TODO: This is very unsafe (as it just casts)
        for (int i = 0; i < this->maxlen; i++) {
            this->data[i] = other[i];
        }
    }

    __host__ __device__ T operator[](int i) const {
        /* Allowing too large `i` for easier handling of different length */
        if (i < this->maxlen) {
            return this->data[i];
        }
        return 0;
    }

    template<typename OT, int Olen>
    __host__ __device__ NumPyString& operator=(const NumPyString<OT, Olen> &other)
    {
        // NOTE: Unlike NumPy, we just cast U->S (unsafe).
        for (int i = 0; i < this->maxlen; i++) {
            this->data[i] = other[i];
        }
        return *this;
    }

    template<typename IntT, typename cuda::std::enable_if<cuda::std::is_integral<IntT>::value, bool>::type = true>
    __host__ __device__ NumPyString& operator=(const IntT &value)
    {
        /* The code below is heavily inspired by the cuDF version */
        char digits[21];  // TODO: 21 is larger than necessary for most ints

        bool const is_negative = value < 0;
        // TODO: The below misbehaves for the -int_min == int_min.
        IntT absval = is_negative ? -value : value;

        int digits_idx = 0;
        int length = maxlen;
        do {
            digits[digits_idx++] = '0' + absval % (IntT)10;
            // next digit
            absval = absval / (IntT)10;
        } while (absval != 0);

        T *ptr = data;
        if (is_negative) {
            *ptr++ = '-';
            length--;
        }
        // digits are backwards, reverse the string into the output
        while (digits_idx-- > 0 && length-- > 0) {
            *ptr++ = digits[digits_idx];
        }

        /* zero fill unused chunk */
        while (length-- > 0) {
            *ptr++ = 0;
        }
        return *this;
    }

    template<typename IntT, typename cuda::std::enable_if<cuda::std::is_integral<IntT>::value, bool>::type = true>
    __host__ __device__ operator IntT()
    {
        /* The code below is heavily inspired by the cuDF version */
        IntT value = 0;

        if (maxlen == 0) {
            return value;
        }
        int length = maxlen;
        T *ptr = data;
        int sign = 1;
        if (*ptr == '-' || *ptr == '+') {
            sign = (*ptr == '-' ? -1 : 1);
            ++ptr;
            --length;
        }
        for (int idx = 0; idx < length; idx++, ptr++) {
            T chr = *ptr;
            if (chr < '0' || chr > '9') {
                // TODO: Maybe it would be good to set INT_MIN/UINT_MAX for real
                //       invalid numbers (at this point could be NULL terminated).
                break;
            }
            value = (value * (IntT)10) + static_cast<IntT>(chr - '0');
        }
        return value * sign;
    }

    template<typename OT, int Olen>
    __host__ __device__ bool operator==(const NumPyString<OT, Olen> &other) const
    {
        int longer = this->maxlen > other.maxlen ? this->maxlen : other.maxlen;
        for (int i = 0; i < longer; i++) {
            if ((*this)[i] != other[i]) {
                return false;
            }
        }
        return true;
    }
};
