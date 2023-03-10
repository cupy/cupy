#pragma once


/* The code below is heavily inspired by the cuDF version */
template<typename IntT, typename CharT>
__host__ __device__ inline void
integer_to_string(const IntT value, int strlen, CharT *ptr_orig)
{
    /* create large enough temporary char string, also to flip later */
    // TODO: 21 is large enough for 64bit integers but this should use
    //       std::numeric_limits or similar?
    char digits[21];

    bool const is_negative = value < 0;
    IntT absval = is_negative ? -value : value;

    int digits_idx = 0;
    do {
        digits[digits_idx++] = '0' + absval % (IntT)10;
        // next digit
        absval = absval / (IntT)10;
    } while (absval != 0);

    CharT *ptr = ptr_orig;
    if (is_negative) {
        *ptr++ = '-';
        strlen--;
    }
    // digits are backwards, reverse the string into the output
    while (digits_idx-- > 0 && strlen-- > 0) {
        *ptr++ = digits[digits_idx];
    }

    /* zero fill unused chunk */
    while (strenlen--) {
        *ptr++ = 0;
    }
}


/* The code below is heavily inspired by the cuDF version */
template<typename IntT, typename CharT>
__host__ __device__ inline IntT
string_to_integer(int strlen, CharT *ptr)
{
    IntT value = 0;

    if (strlen == 0) {
        return value;
    }
    int sign = 1;
    if (*ptr == '-' || *ptr == '+') {
        sign = (*ptr == '-' ? -1 : 1);
        ++ptr;
        --strlen;
    }
    for (int idx = 0; idx < strlen; idx++, ptr++) {
        CharT chr = *ptr;
        if (chr < '0' || chr > '9') {
            // TODO: Maybe it would be good to set INT_MIN/UINT_MAX for real
            //       invalid numbers (at this point could be NULL terminated).
            break;
        }
        value = (value * (IntT)10) + static_cast<IntT>(chr - '0');
    }
    return value * sign;
}


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

    // TODO: Howto best template it for all integers?!
    __host__ __device__ NumPyString& operator=(const long &value)
    {
        integer_to_string(value, maxlen_, this->data);
        return *this;
    }
    __host__ __device__ NumPyString& operator=(const long long &value)
    {
        integer_to_string(value, maxlen_, this->data);
        return *this;
    }


    // TODO: Howto best template it for all integers?!
    __host__ __device__ operator long()
    {
        return string_to_integer<long>(maxlen_, this->data);
    }
    __host__ __device__ operator long long()
    {
        return string_to_integer<long long>(maxlen_, this->data);
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
