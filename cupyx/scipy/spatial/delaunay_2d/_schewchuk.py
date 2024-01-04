
import cupy

SCHEWCHUK_DEF = r"""

*****************************************************************************/
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

"""


SCHEWCHUK_MODULE = cupy.RawModule(
    code=SCHEWCHUK_DEF, options=('-std=c++11',),
    name_expressions=['kerInitPredicate'])


def init_predicate(pred_values):
    ker_init_predicate = SCHEWCHUK_MODULE.get_function('kerInitPredicate')
    ker_init_predicate((1,), (1,), (pred_values))
