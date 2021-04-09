typedef unsigned short npy_uint16;
typedef unsigned int npy_uint32;

/*
 ********************************************************************
 *                     BIT-LEVEL CONVERSIONS                        *
 ********************************************************************
 */


npy_uint16 npy_floatbits_to_halfbits(npy_uint32 f)
{
    npy_uint32 f_exp, f_sig;
    npy_uint16 h_sgn, h_exp, h_sig;

    h_sgn = (npy_uint16) ((f&0x80000000u) >> 16);
    f_exp = (f&0x7f800000u);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /* Inf or NaN */
            f_sig = (f&0x007fffffu);
            if (f_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                npy_uint16 ret = (npy_uint16) (0x7c00u + (f_sig >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (npy_uint16) (h_sgn + 0x7c00u);
            }
        } else {
            /* overflow to signed inf */
            return (npy_uint16) (h_sgn + 0x7c00u);
        }
    }

    /* Exponent underflow converts to a subnormal half or signed zero */
    if (f_exp <= 0x38000000u) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero half-floats.
         */
        if (f_exp < 0x33000000u) {
            return h_sgn;
        }
        /* Make the subnormal significand */
        f_exp >>= 23;
        f_sig = (0x00800000u + (f&0x007fffffu));
        f_sig >>= (113 - f_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
        /*
         * If the last bit in the half significand is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half significand.  In all other cases, we do.
         */
        if ((f_sig&0x00003fffu) != 0x00001000u) {
            f_sig += 0x00001000u;
        }
        h_sig = (npy_uint16) (f_sig >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return (npy_uint16) (h_sgn + h_sig);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (npy_uint16) ((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_sig = (f&0x007fffffu);
    /*
     * If the last bit in the half significand is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half significand.  In all other cases, we do.
     */
    if ((f_sig&0x00003fffu) != 0x00001000u) {
        f_sig += 0x00001000u;
    }
    h_sig = (npy_uint16) (f_sig >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
    return h_sgn + h_exp + h_sig;
}

npy_uint32 npy_halfbits_to_floatbits(npy_uint16 h)
{
    npy_uint16 h_exp, h_sig;
    npy_uint32 f_sgn, f_exp, f_sig;

    h_exp = (h&0x7c00u);
    f_sgn = ((npy_uint32)h&0x8000u) << 16;
    switch (h_exp) {
        case 0x0000u: /* 0 or subnormal */
            h_sig = (h&0x03ffu);
            /* Signed zero */
            if (h_sig == 0) {
                return f_sgn;
            }
            /* Subnormal */
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            f_exp = ((npy_uint32)(127 - 15 - h_exp)) << 23;
            f_sig = ((npy_uint32)(h_sig&0x03ffu)) << 13;
            return f_sgn + f_exp + f_sig;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the significand */
            return f_sgn + 0x7f800000u + (((npy_uint32)(h&0x03ffu)) << 13);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return f_sgn + (((npy_uint32)(h&0x7fffu) + 0x1c000u) << 13);
    }
}
