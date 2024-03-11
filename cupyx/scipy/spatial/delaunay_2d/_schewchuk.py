
SCHEWCHUK_DEF = r"""
#include <cupy/math_constants.h>

/*****************************************************************************/
/*                                                                           */
/*  Routines for Arbitrary Precision Floating-point Arithmetic               */
/*  and Fast Robust Geometric Predicates                                     */
/*  (predicates.c)                                                           */
/*                                                                           */
/*  May 18, 1996                                                             */
/*                                                                           */
/*  Placed in the public domain by                                           */
/*  Jonathan Richard Shewchuk                                                */
/*  School of Computer Science                                               */
/*  Carnegie Mellon University                                               */
/*  5000 Forbes Avenue                                                       */
/*  Pittsburgh, Pennsylvania  15213-3891                                     */
/*  jrs@cs.cmu.edu                                                           */
/*                                                                           */
/*  This file contains C implementation of algorithms for exact addition     */
/*    and multiplication of floating-point numbers, and predicates for       */
/*    robustly performing the orientation and incircle tests used in         */
/*    computational geometry.  The algorithms and underlying theory are      */
/*    described in Jonathan Richard Shewchuk.  "Adaptive Precision Floating- */
/*    Point Arithmetic and Fast Robust Geometric Predicates."  Technical     */
/*    Report CMU-CS-96-140, School of Computer Science, Carnegie Mellon      */
/*    University, Pittsburgh, Pennsylvania, May 1996.  (Submitted to         */
/*    Discrete & Computational Geometry.)                                    */
/*                                                                           */
/*  This file, the paper listed above, and other information are available   */
/*    from the Web page http://www.cs.cmu.edu/~quake/robust.html .           */
/*                                                                           */
/*****************************************************************************/

#define Absolute(a) fabs(a)
#define MUL(a,b)    __dmul_rn(a,b)
using RealType = double;

/* Which of the following two methods of finding the absolute values is      */
/*   fastest is compiler-dependent.  A few compilers can inline and optimize */
/*   the fabs() call; but most will incur the overhead of a function call,   */
/*   which is disastrously slow.  A faster way on IEEE machines might be to  */
/*   mask the appropriate bit, but that's difficult to do in C.              */

//#define Absolute(a)  ((a) >= 0.0 ? (a) : -(a))


/* Many of the operations are broken up into two pieces, a main part that    */
/*   performs an approximate operation, and a "tail" that computes the       */
/*   roundoff error of that operation.                                       */
/*                                                                           */
/* The operations Fast_Two_Sum(), Fast_Two_Diff(), Two_Sum(), Two_Diff(),    */
/*   Split(), and Two_Product() are all implemented as described in the      */
/*   reference.  Each of these macros requires certain variables to be       */
/*   defined in the calling routine.  The variables `bvirt', `c', `abig',    */
/*   `_i', `_j', `_k', `_l', `_m', and `_n' are declared `INEXACT' because   */
/*   they store the result of an operation that may incur roundoff error.    */
/*   The input parameter `x' (or the highest numbered `x_' parameter) must   */
/*   also be declared `INEXACT'.                                             */

#define Fast_Two_Sum_Tail(a, b, x, y) \
    bvirt = x - a; \
    y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
    x = (RealType) (a + b); \
    Fast_Two_Sum_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y) \
    bvirt = (RealType) (x - a); \
    avirt = x - bvirt; \
    bround = b - bvirt; \
    around = a - avirt; \
    y = around + bround

#define Two_Sum(a, b, x, y) \
    x = (RealType) (a + b); \
    Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y) \
    bvirt = (RealType) (a - x); \
    avirt = x + bvirt; \
    bround = bvirt - b; \
    around = a - avirt; \
    y = around + bround

#define Two_Diff(a, b, x, y) \
    x = (RealType) (a - b); \
    Two_Diff_Tail(a, b, x, y)

#define Split(a, ahi, alo) \
    c = MUL(predConsts[ Splitter ], a); \
    abig = (RealType) (c - a); \
    ahi = c - abig; \
    alo = a - ahi

#define Two_Product_Tail(a, b, x, y) \
    Split(a, ahi, alo); \
    Split(b, bhi, blo); \
    err1 = x - MUL(ahi, bhi); \
    err2 = err1 - MUL(alo, bhi); \
    err3 = err2 - MUL(ahi, blo); \
    y = MUL(alo, blo) - err3

#define Two_Product(a, b, x, y) \
    x = MUL(a, b); \
    Two_Product_Tail(a, b, x, y)

/* Two_Product_Presplit() is Two_Product() where one of the inputs has       */
/*   already been split.  Avoids redundant splitting.                        */

#define Two_Product_Presplit(a, b, bhi, blo, x, y) \
    x = MUL(a, b); \
    Split(a, ahi, alo); \
    err1 = x - MUL(ahi, bhi); \
    err2 = err1 - MUL(alo, bhi); \
    err3 = err2 - MUL(ahi, blo); \
    y = MUL(alo, blo) - err3

/* Macros for summing expansions of various fixed lengths.  These are all    */
/*   unrolled versions of Expansion_Sum().                                   */

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
    Two_Diff(a0, b , _i, x0); \
    Two_Sum( a1, _i, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
    Two_One_Diff(a1, a0, b0, _j, _0, x0); \
    Two_One_Diff(_j, _0, b1, x3, x2, x1)

/* Macros for multiplying expansions of various fixed lengths.               */

#define Two_One_Product(a1, a0, b, x3, x2, x1, x0) \
    Split(b, bhi, blo); \
    Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
    Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
    Two_Sum(_i, _0, _k, x1); \
    Fast_Two_Sum(_j, _k, x3, x2)

/*****************************************************************************/
/*                                                                           */
/*  exactinit()   Initialize the variables used for exact arithmetic.        */
/*                                                                           */
/*  `epsilon' is the largest power of two such that 1.0 + epsilon = 1.0 in   */
/*  floating-point arithmetic.  `epsilon' bounds the relative roundoff       */
/*  error.  It is used for floating-point error analysis.                    */
/*                                                                           */
/*  `splitter' is used to split floating-point numbers into two half-        */
/*  length significands for exact multiplication.                            */
/*                                                                           */
/*  I imagine that a highly optimizing compiler might be too smart for its   */
/*  own good, and somehow cause this routine to fail, if it pretends that    */
/*  floating-point arithmetic is too much like real arithmetic.              */
/*                                                                           */
/*  Don't change this routine unless you fully understand it.                */
/*                                                                           */
/*****************************************************************************/


enum DPredicateBounds
{
    Splitter,       /* = 2^ceiling(p / 2) + 1.  Used to split floats in half.*/
    Epsilon,        /* = 2^(-p).  Used to estimate roundoff errors. */

    /* A set of coefficients used to calculate maximum roundoff errors.      */
    Resulterrbound,
    CcwerrboundA,
    CcwerrboundB,
    CcwerrboundC,
    O3derrboundA,
    O3derrboundB,
    O3derrboundC,
    IccerrboundA,
    IccerrboundB,
    IccerrboundC,
    IsperrboundA,
    IsperrboundB,
    IsperrboundC,
    O3derrboundAlifted,
    O2derrboundAlifted,
    O1derrboundAlifted,

    DPredicateBoundNum  // Number of bounds in this enum
};

/*****************************************************************************/
/*                                                                           */
/*  d_scale_expansion_zeroelim()   Multiply an expansion by a scalar,        */
/*                               eliminating zero components from the        */
/*                               output expansion.                           */
/*                                                                           */
/*  Sets h = be.  See either version of my paper for details.                */
/*                                                                           */
/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
/*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    */
/*  properties as well.  (That is, if e has one of these properties, so      */
/*  will h.)                                                                 */
/*                                                                           */
/*****************************************************************************/

/* e and h cannot be the same. */
__device__ int d_scale_expansion_zeroelim
(
const RealType* predConsts,
int             elen,
RealType*       e,
RealType        b,
RealType*       h
)
{
    RealType Q, sum;
    RealType hh;
    RealType product1;
    RealType product0;
    int eindex, hindex;
    RealType enow;
    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;

    Split(b, bhi, blo);
    Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
    hindex = 0;
    if (hh != 0) {
        h[hindex++] = hh;
    }
    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
        Two_Sum(Q, product0, sum, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
        Fast_Two_Sum(product1, sum, Q, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

/*****************************************************************************/
/*                                                                           */
/*  d_fast_expansion_sum_zeroelim()   Sum two expansions, eliminating zero   */
/*                                  components from the output expansion.    */
/*                                                                           */
/*  Sets h = e + f.  See the long version of my paper for details.           */
/*                                                                           */
/*  If round-to-even is used (as with IEEE 754), maintains the strongly      */
/*  nonoverlapping property.  (That is, if e is strongly nonoverlapping, h   */
/*  will be also.)  Does NOT maintain the nonoverlapping or nonadjacent      */
/*  properties.                                                              */
/*                                                                           */
/*****************************************************************************/

/* h cannot be e or f. */
__device__ int d_fast_expansion_sum_zeroelim
(
int      elen,
RealType *e,
int      flen,
RealType *f,
RealType *h
)
{
    RealType Q;
    RealType Qnew;
    RealType hh;
    RealType bvirt;
    RealType avirt, bround, around;
    int eindex, findex, hindex;
    RealType enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        enow = e[++eindex];
    } else {
        Q = fnow;
        fnow = f[++findex];
    }
    hindex = 0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            enow = e[++eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            fnow = f[++findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                enow = e[++eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                fnow = f[++findex];
            }
            Q = Qnew;
            if (hh != 0.0) {
                h[hindex++] = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

__device__ RealType d_fast_expansion_sum_sign
(
int      elen,
RealType *e,
int      flen,
RealType *f
)
{
    RealType Q;
    RealType lastTerm;
    RealType Qnew;
    RealType hh;
    RealType bvirt;
    RealType avirt, bround, around;
    int eindex, findex;
    RealType enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        enow = e[++eindex];
    } else {
        Q = fnow;
        fnow = f[++findex];
    }
    lastTerm = 0.0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            enow = e[++eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            fnow = f[++findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                enow = e[++eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                fnow = f[++findex];
            }
            Q = Qnew;
            if (hh != 0.0) {
                lastTerm = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
    }
    if ( Q != 0.0 ) {
        lastTerm = Q;
    }
    return lastTerm;
}

__device__ int d_scale_twice_expansion_zeroelim
(
const RealType* predConsts,
int elen,
RealType *e,
RealType b1,
RealType b2,
RealType *h
)
{
    RealType Q, sum, Q2, sum2;
    RealType hh;
    RealType product1, product2;
    RealType product0;
    int eindex, hindex;
    RealType enow;
    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, b1hi, b1lo, b2hi, b2lo;
    RealType err1, err2, err3;

    hindex = 0;

    Split(b1, b1hi, b1lo);
    Split(b2, b2hi, b2lo);
    Two_Product_Presplit(e[0], b1, b1hi, b1lo, Q, hh);
    Two_Product_Presplit(hh, b2, b2hi, b2lo, Q2, hh);

    if (hh != 0) {
        h[hindex++] = hh;
    }

    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product_Presplit(enow, b1, b1hi, b1lo, product1, product0);
        Two_Sum(Q, product0, sum, hh);

        Two_Product_Presplit(hh, b2, b2hi, b2lo, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product1, sum, Q, hh);

        Two_Product_Presplit(hh, b2, b2hi, b2lo, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }

    if (Q != 0) {
        Two_Product_Presplit(Q, b2, b2hi, b2lo, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);

        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }

    if ((Q2 != 0) || (hindex == 0)) {
        h[hindex++] = Q2;
    }

    return hindex;
}

__global__ void kerInitPredicate( RealType* predConsts )
{
    RealType half;
    RealType epsilon, splitter;
    RealType check, lastcheck;
    int every_other;

    every_other = 1;
    half        = 0.5;
    epsilon     = 1.0;
    splitter    = 1.0;
    check       = 1.0;

    /* Repeatedly divide `epsilon' by two until it is too small to add to    */
    /*   one without causing roundoff.  (Also check if the sum is equal to   */
    /*   the previous sum, for machines that round up instead of using exact */
    /*   rounding.  Not that this library will work on such machines anyway. */
    do
    {
        lastcheck   = check;
        epsilon     *= half;

        if (every_other)
        {
            splitter *= 2.0;
        }

        every_other = !every_other;
        check       = 1.0 + epsilon;
    } while ((check != 1.0) && (check != lastcheck));

    /* Error bounds for orientation and incircle tests. */
    predConsts[ Epsilon ]           = epsilon;
    predConsts[ Splitter ]          = splitter + 1.0;
    predConsts[ Resulterrbound ]    = (3.0 + 8.0 * epsilon) * epsilon;
    predConsts[ CcwerrboundA ]      = (3.0 + 16.0 * epsilon) * epsilon;
    predConsts[ CcwerrboundB ]      = (2.0 + 12.0 * epsilon) * epsilon;
    predConsts[ CcwerrboundC ]    = (9.0 + 64.0 * epsilon) * epsilon * epsilon;
    predConsts[ O3derrboundA ]      = (7.0 + 56.0 * epsilon) * epsilon;
    predConsts[ O3derrboundB ]      = (3.0 + 28.0 * epsilon) * epsilon;
    predConsts[ O3derrboundC ]  = (26.0 + 288.0 * epsilon) * epsilon * epsilon;
    predConsts[ IccerrboundA ]      = (10.0 + 96.0 * epsilon) * epsilon;
    predConsts[ IccerrboundB ]      = (4.0 + 48.0 * epsilon) * epsilon;
    predConsts[ IccerrboundC ]  = (44.0 + 576.0 * epsilon) * epsilon * epsilon;
    predConsts[ IsperrboundA ]      = (16.0 + 224.0 * epsilon) * epsilon;
    predConsts[ IsperrboundB ]      = (5.0 + 72.0 * epsilon) * epsilon;
    predConsts[ IsperrboundC ] = (71.0 + 1408.0 * epsilon) * epsilon * epsilon;
    predConsts[ O3derrboundAlifted ] = (11.0 + 112.0 * epsilon) * epsilon;
        //(10.0 + 112.0 * epsilon) * epsilon;
    predConsts[ O2derrboundAlifted ]= (6.0 + 48.0 * epsilon) * epsilon;
    predConsts[ O1derrboundAlifted ]= (3.0 + 16.0 * epsilon) * epsilon;

    return;
}

__forceinline__ __device__ RealType orient2dExact
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc
)
{
    RealType axby1, axcy1, bxcy1, bxay1, cxay1, cxby1;
    RealType axby0, axcy0, bxcy0, bxay0, cxay0, cxby0;
    RealType aterms[4], bterms[4], cterms[4];
    RealType v[8];
    int vlength;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    Two_Product(pa[0], pb[1], axby1, axby0);
    Two_Product(pa[0], pc[1], axcy1, axcy0);
    Two_Two_Diff(axby1, axby0, axcy1, axcy0,
        aterms[3], aterms[2], aterms[1], aterms[0]);

    Two_Product(pb[0], pc[1], bxcy1, bxcy0);
    Two_Product(pb[0], pa[1], bxay1, bxay0);
    Two_Two_Diff(bxcy1, bxcy0, bxay1, bxay0,
        bterms[3], bterms[2], bterms[1], bterms[0]);

    Two_Product(pc[0], pa[1], cxay1, cxay0);
    Two_Product(pc[0], pb[1], cxby1, cxby0);
    Two_Two_Diff(cxay1, cxay0, cxby1, cxby0,
        cterms[3], cterms[2], cterms[1], cterms[0]);

    vlength = d_fast_expansion_sum_zeroelim(4, aterms, 4, bterms, v);

    return d_fast_expansion_sum_sign(vlength, v, 4, cterms);
}

__device__ RealType orient2dFast
(
const RealType* predConsts,
const RealType *pa,
const RealType *pb,
const RealType *pc
)
{
    RealType detleft, detright, det;
    RealType detsum, errbound;

    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    det = detleft - detright;

    if (detleft > 0.0) {
        if (detright <= 0.0) {
            return det;
        } else {
            detsum = detleft + detright;
        }
    } else if (detleft < 0.0) {
        if (detright >= 0.0) {
            return det;
        } else {
            detsum = -detleft - detright;
        }
    } else {
        return det;
    }

    errbound = predConsts[ CcwerrboundA ] * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    return 0.0;
}

__device__ RealType orient2dFastExact
(
const RealType* predConsts,
const RealType *pa,
const RealType *pb,
const RealType *pc
)
{
    RealType detleft, detright, det;
    RealType detsum, errbound;

    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    det = detleft - detright;

    if (detleft > 0.0) {
        if (detright <= 0.0) {
            return det;
        } else {
            detsum = detleft + detright;
        }
    } else if (detleft < 0.0) {
        if (detright >= 0.0) {
            return det;
        } else {
            detsum = -detleft - detright;
        }
    } else {
        return det;
    }

    errbound = predConsts[ CcwerrboundA ] * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    return orient2dExact( predConsts, pa, pb, pc );
}

__forceinline__ __device__ void two_mult_sub
(
const RealType *predConsts,
const RealType *pa,
const RealType *pb,
RealType       *ab
)
{
    RealType axby1, axby0, bxay1, bxay0;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    Two_Product(pa[0], pb[1], axby1, axby0);
    Two_Product(pb[0], pa[1], bxay1, bxay0);
    Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);
}


__noinline__ __device__ int calc_det
(
const RealType  *predConsts,
RealType        *a,
RealType        *b,
RealType        *c,
RealType        fx0,
RealType        fx1,
RealType        fy0,
RealType        fy1,
RealType        *temp2,
RealType        *temp3,
RealType        *detx,
RealType        *dety,
RealType        *ret
)
{
    int temp2len = d_fast_expansion_sum_zeroelim( 4, a, 4, b, temp2 );
    int temp3len = d_fast_expansion_sum_zeroelim(
        temp2len, temp2, 4, c, temp3 );

    int xlen = d_scale_twice_expansion_zeroelim(
        predConsts, temp3len, temp3, fx0, fx1, detx );
    int ylen = d_scale_twice_expansion_zeroelim(
        predConsts, temp3len, temp3, fy0, fy1, dety );

    return d_fast_expansion_sum_zeroelim( xlen, detx, ylen, dety, ret );
}

__device__ RealType incircleExact
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd
)
{
    RealType ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
    RealType temp8[8];
    RealType temp12[12];
    RealType det48x[48], det48y[48];
    RealType bdet[96], cdet[96];
    int alen, blen, clen, dlen;
    RealType bcdet[192], addet[192];
    int bclen, adlen;
    int i;

    RealType *adet = bdet;
    RealType *ddet = cdet;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    two_mult_sub( predConsts, pa, pb, ab );
    two_mult_sub( predConsts, pb, pc, bc );
    two_mult_sub( predConsts, pc, pd, cd );
    two_mult_sub( predConsts, pd, pa, da );
    two_mult_sub( predConsts, pa, pc, ac );
    two_mult_sub( predConsts, pb, pd, bd );

    blen = calc_det( predConsts, cd, da, ac, pb[0], -pb[0], pb[1], -pb[1],
        temp8, temp12, det48x, det48y, bdet );
    clen = calc_det( predConsts, da, ab, bd, pc[0], pc[0], pc[1], pc[1],
        temp8, temp12, det48x, det48y, cdet );

    bclen = d_fast_expansion_sum_zeroelim(blen, bdet, clen, cdet, bcdet);

    for (i = 0; i < 4; i++) {
        bd[i] = -bd[i];
        ac[i] = -ac[i];
    }

    dlen = calc_det( predConsts, ab, bc, ac, pd[0], -pd[0], pd[1], -pd[1],
        temp8, temp12, det48x, det48y, ddet );
    alen = calc_det( predConsts, bc, cd, bd, pa[0], pa[0], pa[1], pa[1],
        temp8, temp12, det48x, det48y, adet );

    adlen = d_fast_expansion_sum_zeroelim(alen, adet, dlen, ddet, addet);

    return d_fast_expansion_sum_sign( bclen, bcdet, adlen, addet );
}

__noinline__ __device__ int calc_det_adapt
(
const RealType  *predConsts,
RealType        adx,
RealType        ady,
RealType        bdx,
RealType        bdy,
RealType        cdx,
RealType        cdy,
RealType        *temp4,
RealType        *temp16x,
RealType        *temp16y,
RealType        *ret
)
{
    RealType axby1, axby0, bxay1, bxay0;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    Two_Product(adx, bdy, axby1, axby0);
    Two_Product(bdx, ady, bxay1, bxay0);
    Two_Two_Diff(
        axby1, axby0, bxay1, bxay0, temp4[3], temp4[2], temp4[1], temp4[0]);

    int temp16xlen = d_scale_twice_expansion_zeroelim(
        predConsts, 4, temp4, cdx, cdx, temp16x);
    int temp16ylen = d_scale_twice_expansion_zeroelim(
        predConsts, 4, temp4, cdy, cdy, temp16y);

    return d_fast_expansion_sum_zeroelim(
        temp16xlen, temp16x, temp16ylen, temp16y, ret );
}

__device__ RealType incircleAdaptExact
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd,
const RealType  permanent
)
{
    RealType adx, bdx, cdx, ady, bdy, cdy;
    RealType det, errbound;

    RealType temp4[4];
    RealType temp16x[16], temp16y[16];
    RealType adet[32], bdet[32];
    int alen, blen, clen;
    RealType abdet[64];
    int ablen;

    RealType* cdet = adet;

    RealType adxtail, bdxtail, cdxtail, adytail, bdytail, cdytail;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    adx = (RealType) (pa[0] - pd[0]);
    bdx = (RealType) (pb[0] - pd[0]);
    cdx = (RealType) (pc[0] - pd[0]);
    ady = (RealType) (pa[1] - pd[1]);
    bdy = (RealType) (pb[1] - pd[1]);
    cdy = (RealType) (pc[1] - pd[1]);

    alen = calc_det_adapt(
        predConsts, bdx, bdy, cdx, cdy, adx, ady, temp4, temp16x,
        temp16y, adet );
    blen = calc_det_adapt( predConsts, cdx, cdy, adx, ady, bdx, bdy,
                           temp4, temp16x, temp16y, bdet );

    ablen = d_fast_expansion_sum_zeroelim( alen, adet, blen, bdet, abdet );

    clen = calc_det_adapt( predConsts, adx, ady, bdx, bdy, cdx, cdy,
                           temp4, temp16x, temp16y, cdet );

    det = d_fast_expansion_sum_sign( ablen, abdet, clen, cdet );

    errbound = predConsts[ IccerrboundB ] * permanent;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Diff_Tail(pa[0], pd[0], adx, adxtail);
    Two_Diff_Tail(pa[1], pd[1], ady, adytail);
    Two_Diff_Tail(pb[0], pd[0], bdx, bdxtail);
    Two_Diff_Tail(pb[1], pd[1], bdy, bdytail);
    Two_Diff_Tail(pc[0], pd[0], cdx, cdxtail);
    Two_Diff_Tail(pc[1], pd[1], cdy, cdytail);
    if ((adxtail == 0.0) && (bdxtail == 0.0) && (cdxtail == 0.0)
        && (adytail == 0.0) && (bdytail == 0.0) && (cdytail == 0.0)) {
            return det;
    }

    errbound = ( predConsts[ IccerrboundC ] * permanent +
                 predConsts[ Resulterrbound ] * Absolute(det) );
    det += ((adx * adx + ady * ady) * ((bdx * cdytail + cdy * bdxtail)
        - (bdy * cdxtail + cdx * bdytail))
        + 2.0 * (adx * adxtail + ady * adytail) * (bdx * cdy - bdy * cdx))
        + ((bdx * bdx + bdy * bdy) * ((cdx * adytail + ady * cdxtail)
        - (cdy * adxtail + adx * cdytail))
        + 2.0 * (bdx * bdxtail + bdy * bdytail) * (cdx * ady - cdy * adx))
        + ((cdx * cdx + cdy * cdy) * ((adx * bdytail + bdy * adxtail)
        - (ady * bdxtail + bdx * adytail))
        + 2.0 * (cdx * cdxtail + cdy * cdytail) * (adx * bdy - ady * bdx));
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    return incircleExact( predConsts, pa, pb, pc, pd );
}

__forceinline__ __device__ RealType incircleFastAdaptExact
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd
)
{
    RealType adx, bdx, cdx, ady, bdy, cdy;
    RealType bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    RealType alift, blift, clift;
    RealType det;
    RealType permanent, errbound;

    adx = pa[0] - pd[0];
    bdx = pb[0] - pd[0];
    cdx = pc[0] - pd[0];
    ady = pa[1] - pd[1];
    bdy = pb[1] - pd[1];
    cdy = pc[1] - pd[1];

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;
    alift = adx * adx + ady * ady;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;
    blift = bdx * bdx + bdy * bdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;
    clift = cdx * cdx + cdy * cdy;

    det = alift * (bdxcdy - cdxbdy)
        + blift * (cdxady - adxcdy)
        + clift * (adxbdy - bdxady);

    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * alift
        + (Absolute(cdxady) + Absolute(adxcdy)) * blift
        + (Absolute(adxbdy) + Absolute(bdxady)) * clift;
    errbound = predConsts[ IccerrboundA ] * permanent;
    if ((det > errbound) || (-det > errbound)) {
        return det;
    }

    // Needs exact predicate
    return incircleAdaptExact( predConsts, pa, pb, pc, pd, permanent );
}

__forceinline__ __device__ RealType incircleFast
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd
)
{
    RealType adx, bdx, cdx, ady, bdy, cdy;
    RealType bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    RealType alift, blift, clift;
    RealType det;
    RealType permanent, errbound;

    adx = pa[0] - pd[0];
    bdx = pb[0] - pd[0];
    cdx = pc[0] - pd[0];
    ady = pa[1] - pd[1];
    bdy = pb[1] - pd[1];
    cdy = pc[1] - pd[1];

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;
    alift = adx * adx + ady * ady;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;
    blift = bdx * bdx + bdy * bdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;
    clift = cdx * cdx + cdy * cdy;

    det = alift * (bdxcdy - cdxbdy)
        + blift * (cdxady - adxcdy)
        + clift * (adxbdy - bdxady);

    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * alift
        + (Absolute(cdxady) + Absolute(adxcdy)) * blift
        + (Absolute(adxbdy) + Absolute(bdxady)) * clift;
    errbound = predConsts[ IccerrboundA ] * permanent;
    if ((det > errbound) || (-det > errbound)) {
        return det;
    }

    return 0;   // Needs exact predicate
}

///////////////////////////////////////////////////////////////////////////////

// det  = ( pa[0]^2 + pa[1]^2 ) - ( pb[0]^2 + pb[1]^2 )
__device__ RealType orient1dExact_Lifted
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb
)
{
    RealType axax1, ayay1, bxbx1, byby1;
    RealType axax0, ayay0, bxbx0, byby0;
    RealType aterms[4], bterms[4];

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    Two_Product(pa[0], pa[0], axax1, axax0);
    Two_Product(pb[0], pb[0], bxbx1, bxbx0);
    Two_Two_Diff(axax1, axax0, bxbx1, bxbx0,
        aterms[3], aterms[2], aterms[1], aterms[0]);

    Two_Product(pa[1], pa[1], ayay1, ayay0);
    Two_Product(pb[1], pb[1], byby1, byby0);
    Two_Two_Diff(ayay1, ayay0, byby1, byby0,
        bterms[3], bterms[2], bterms[1], bterms[0]);

    return d_fast_expansion_sum_sign(4, aterms, 4, bterms);
}

__device__ RealType orient2dExact_Lifted
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc,
bool            lifted
)
{
    RealType aax1, aax0, aay1, aay0;
    RealType palift[4], pblift[4], pclift[4];
    RealType xy1terms[8], xy2terms[8];
    RealType aterms[16], bterms[16];
    RealType v[32];

    RealType* cterms = aterms;

    int palen, pblen, pclen;
    int xy1len, xy2len;
    int alen, blen, clen;
    int vlen;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    // Compute the lifted coordinate
    if (lifted)
    {
        Two_Product(pa[0], pa[0], aax1, aax0);
        Two_Product(-pa[1], pa[1], aay1, aay0);
        Two_Two_Diff(
            aax1, aax0, aay1, aay0, palift[3], palift[2],
            palift[1], palift[0]);
        palen = 4;

        Two_Product(pb[0], pb[0], aax1, aax0);
        Two_Product(-pb[1], pb[1], aay1, aay0);
        Two_Two_Diff(
            aax1, aax0, aay1, aay0, pblift[3], pblift[2],
            pblift[1], pblift[0]);
        pblen = 4;

        Two_Product(pc[0], pc[0], aax1, aax0);
        Two_Product(-pc[1], pc[1], aay1, aay0);
        Two_Two_Diff(
            aax1, aax0, aay1, aay0, pclift[3], pclift[2],
            pclift[1], pclift[0]);
        pclen = 4;
    }
    else {
        palen = 1; palift[0] = pa[1];
        pblen = 1; pblift[0] = pb[1];
        pclen = 1; pclift[0] = pc[1];
    }

    // Compute the determinant as usual
    xy1len = d_scale_expansion_zeroelim(
        predConsts, pblen, pblift, pa[0], xy1terms);
    xy2len = d_scale_expansion_zeroelim(
        predConsts, pclen, pclift, -pa[0], xy2terms);
    alen = d_fast_expansion_sum_zeroelim(
        xy1len, xy1terms, xy2len, xy2terms, aterms);

    xy1len = d_scale_expansion_zeroelim(
        predConsts, pclen, pclift, pb[0], xy1terms);
    xy2len = d_scale_expansion_zeroelim(
        predConsts, palen, palift, -pb[0], xy2terms);
    blen = d_fast_expansion_sum_zeroelim(
        xy1len, xy1terms, xy2len, xy2terms, bterms);

    vlen = d_fast_expansion_sum_zeroelim(alen, aterms, blen, bterms, v);

    xy1len = d_scale_expansion_zeroelim(
        predConsts, palen, palift, pc[0], xy1terms);
    xy2len = d_scale_expansion_zeroelim(
        predConsts, pblen, pblift, -pc[0], xy2terms);
    clen = d_fast_expansion_sum_zeroelim(
        xy1len, xy1terms, xy2len, xy2terms, cterms);

    return d_fast_expansion_sum_sign(vlen, v, clen, cterms);
}

"""
