# This source code contains SciPy's code.
# https://github.com/scipy/scipy/blob/master/scipy/special/cephes/k0.c
# https://github.com/scipy/scipy/blob/master/scipy/special/cephes/k1.c
#
#
# Cephes Math Library Release 2.8:  June, 2000
# Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier

from cupy import _core
from cupyx.scipy.special._gamma import chbevl_implementation

j0 = _core.create_ufunc(
    'cupyx_scipy_special_j0', ('f->f', 'd->d'),
    'out0 = j0(in0)',
    doc='''Bessel function of the first kind of order 0.

    .. seealso:: :meth:`scipy.special.j0`

    ''')


j1 = _core.create_ufunc(
    'cupyx_scipy_special_j1', ('f->f', 'd->d'),
    'out0 = j1(in0)',
    doc='''Bessel function of the first kind of order 1.

    .. seealso:: :meth:`scipy.special.j1`

    ''')


y0 = _core.create_ufunc(
    'cupyx_scipy_special_y0', ('f->f', 'd->d'),
    'out0 = y0(in0)',
    doc='''Bessel function of the second kind of order 0.

    .. seealso:: :meth:`scipy.special.y0`

    ''')


y1 = _core.create_ufunc(
    'cupyx_scipy_special_y1', ('f->f', 'd->d'),
    'out0 = y1(in0)',
    doc='''Bessel function of the second kind of order 1.

    .. seealso:: :meth:`scipy.special.y1`

    ''')


# Note: oddly, unlike for y0 or y1, SciPy always returns double for yn
# dd->d because SciPy will accept 2.0
yn = _core.create_ufunc(
    'cupyx_scipy_special_yn', ('id->d', 'dd->d'),
    'out0 = yn((int)in0, in1)',
    doc='''Bessel function of the second kind of order n.

    Args:
        n (cupy.ndarray): order (integer)
        x (cupy.ndarray): argument (float)

    Returns:
        cupy.ndarray: The result.

    Notes
    -----
    Unlike SciPy, no warning will be raised on unsafe casting of `order` to
    32-bit integer.

    .. seealso:: :meth:`scipy.special.yn`
    ''')


i0 = _core.create_ufunc(
    'cupyx_scipy_special_i0', ('f->f', 'd->d'),
    'out0 = cyl_bessel_i0(in0)',
    doc='''Modified Bessel function of order 0.

    .. seealso:: :meth:`scipy.special.i0`

    ''')


i0e = _core.create_ufunc(
    'cupyx_scipy_special_i0e', ('f->f', 'd->d'),
    'out0 = exp(-abs(in0)) * cyl_bessel_i0(in0)',
    doc='''Exponentially scaled modified Bessel function of order 0.

    .. seealso:: :meth:`scipy.special.i0e`

    ''')


i1 = _core.create_ufunc(
    'cupyx_scipy_special_i1', ('f->f', 'd->d'),
    'out0 = cyl_bessel_i1(in0)',
    doc='''Modified Bessel function of order 1.

    .. seealso:: :meth:`scipy.special.i1`

    ''')


i1e = _core.create_ufunc(
    'cupyx_scipy_special_i1e', ('f->f', 'd->d'),
    'out0 = exp(-abs(in0)) * cyl_bessel_i1(in0)',
    doc='''Exponentially scaled modified Bessel function of order 1.

    .. seealso:: :meth:`scipy.special.i1e`

    ''')


k_preamble = chbevl_implementation + """
#include <cupy/math_constants.h>

/* Chebyshev coefficients for K0(x) + log(x/2) I0(x)
 * in the interval [0,2].  The odd order coefficients are all
 * zero; only the even order coefficients are listed.
 *
 * lim(x->0){ K0(x) + log(x/2) I0(x) } = -EUL.
 */
__device__ double k0_A[] = {
    1.37446543561352307156E-16,
    4.25981614279661018399E-14,
    1.03496952576338420167E-11,
    1.90451637722020886025E-9,
    2.53479107902614945675E-7,
    2.28621210311945178607E-5,
    1.26461541144692592338E-3,
    3.59799365153615016266E-2,
    3.44289899924628486886E-1,
    -5.35327393233902768720E-1
};
__device__ float k0_AF[] = {
    1.37446543561352307156E-16,
    4.25981614279661018399E-14,
    1.03496952576338420167E-11,
    1.90451637722020886025E-9,
    2.53479107902614945675E-7,
    2.28621210311945178607E-5,
    1.26461541144692592338E-3,
    3.59799365153615016266E-2,
    3.44289899924628486886E-1,
    -5.35327393233902768720E-1
};


/* Chebyshev coefficients for exp(x) sqrt(x) K0(x)
 * in the inverted interval [2,infinity].
 *
 * lim(x->inf){ exp(x) sqrt(x) K0(x) } = sqrt(pi/2).
 */
__device__ double k0_B[] = {
    5.30043377268626276149E-18,
    -1.64758043015242134646E-17,
    5.21039150503902756861E-17,
    -1.67823109680541210385E-16,
    5.51205597852431940784E-16,
    -1.84859337734377901440E-15,
    6.34007647740507060557E-15,
    -2.22751332699166985548E-14,
    8.03289077536357521100E-14,
    -2.98009692317273043925E-13,
    1.14034058820847496303E-12,
    -4.51459788337394416547E-12,
    1.85594911495471785253E-11,
    -7.95748924447710747776E-11,
    3.57739728140030116597E-10,
    -1.69753450938905987466E-9,
    8.57403401741422608519E-9,
    -4.66048989768794782956E-8,
    2.76681363944501510342E-7,
    -1.83175552271911948767E-6,
    1.39498137188764993662E-5,
    -1.28495495816278026384E-4,
    1.56988388573005337491E-3,
    -3.14481013119645005427E-2,
    2.44030308206595545468E0
};
__device__ float k0_BF[] = {
    5.30043377268626276149E-18,
    -1.64758043015242134646E-17,
    5.21039150503902756861E-17,
    -1.67823109680541210385E-16,
    5.51205597852431940784E-16,
    -1.84859337734377901440E-15,
    6.34007647740507060557E-15,
    -2.22751332699166985548E-14,
    8.03289077536357521100E-14,
    -2.98009692317273043925E-13,
    1.14034058820847496303E-12,
    -4.51459788337394416547E-12,
    1.85594911495471785253E-11,
    -7.95748924447710747776E-11,
    3.57739728140030116597E-10,
    -1.69753450938905987466E-9,
    8.57403401741422608519E-9,
    -4.66048989768794782956E-8,
    2.76681363944501510342E-7,
    -1.83175552271911948767E-6,
    1.39498137188764993662E-5,
    -1.28495495816278026384E-4,
    1.56988388573005337491E-3,
    -3.14481013119645005427E-2,
    2.44030308206595545468E0
};


/* Chebyshev coefficients for x(K1(x) - log(x/2) I1(x))
 * in the interval [0,2].
 *
 * lim(x->0){ x(K1(x) - log(x/2) I1(x)) } = 1.
 */
__device__ static double k1_A[] = {
    -7.02386347938628759343E-18,
    -2.42744985051936593393E-15,
    -6.66690169419932900609E-13,
    -1.41148839263352776110E-10,
    -2.21338763073472585583E-8,
    -2.43340614156596823496E-6,
    -1.73028895751305206302E-4,
    -6.97572385963986435018E-3,
    -1.22611180822657148235E-1,
    -3.53155960776544875667E-1,
    1.52530022733894777053E0
};
__device__ static float k1_AF[] = {
    -7.02386347938628759343E-18,
    -2.42744985051936593393E-15,
    -6.66690169419932900609E-13,
    -1.41148839263352776110E-10,
    -2.21338763073472585583E-8,
    -2.43340614156596823496E-6,
    -1.73028895751305206302E-4,
    -6.97572385963986435018E-3,
    -1.22611180822657148235E-1,
    -3.53155960776544875667E-1,
    1.52530022733894777053E0
};


/* Chebyshev coefficients for exp(x) sqrt(x) K1(x)
 * in the interval [2,infinity].
 *
 * lim(x->inf){ exp(x) sqrt(x) K1(x) } = sqrt(pi/2).
 */
__device__ static double k1_B[] = {
    -5.75674448366501715755E-18,
    1.79405087314755922667E-17,
    -5.68946255844285935196E-17,
    1.83809354436663880070E-16,
    -6.05704724837331885336E-16,
    2.03870316562433424052E-15,
    -7.01983709041831346144E-15,
    2.47715442448130437068E-14,
    -8.97670518232499435011E-14,
    3.34841966607842919884E-13,
    -1.28917396095102890680E-12,
    5.13963967348173025100E-12,
    -2.12996783842756842877E-11,
    9.21831518760500529508E-11,
    -4.19035475934189648750E-10,
    2.01504975519703286596E-9,
    -1.03457624656780970260E-8,
    5.74108412545004946722E-8,
    -3.50196060308781257119E-7,
    2.40648494783721712015E-6,
    -1.93619797416608296024E-5,
    1.95215518471351631108E-4,
    -2.85781685962277938680E-3,
    1.03923736576817238437E-1,
    2.72062619048444266945E0
};
__device__ static float k1_BF[] = {
    -5.75674448366501715755E-18,
    1.79405087314755922667E-17,
    -5.68946255844285935196E-17,
    1.83809354436663880070E-16,
    -6.05704724837331885336E-16,
    2.03870316562433424052E-15,
    -7.01983709041831346144E-15,
    2.47715442448130437068E-14,
    -8.97670518232499435011E-14,
    3.34841966607842919884E-13,
    -1.28917396095102890680E-12,
    5.13963967348173025100E-12,
    -2.12996783842756842877E-11,
    9.21831518760500529508E-11,
    -4.19035475934189648750E-10,
    2.01504975519703286596E-9,
    -1.03457624656780970260E-8,
    5.74108412545004946722E-8,
    -3.50196060308781257119E-7,
    2.40648494783721712015E-6,
    -1.93619797416608296024E-5,
    1.95215518471351631108E-4,
    -2.85781685962277938680E-3,
    1.03923736576817238437E-1,
    2.72062619048444266945E0
};

__device__ double k0(double x){
    if (x == 0) {
        return CUDART_INF;
    }

    if (x < 0) {
        return CUDART_NAN;
    }

    double y, z;

    if (x <= 2.0) {
        y = x * x - 2.0;
        y = chbevl(y, k0_A, 10) - log(0.5 * x) * cyl_bessel_i0(x);
        return y;
    }

    z = 8.0 / x - 2.0;
    y = chbevl(z, k0_B, 25) / sqrt(x);
    return y * exp(-x);
}

__device__ float k0f(float x){
    if (x == 0) {
        return CUDART_INF;
    }

    if (x < 0) {
        return CUDART_NAN;
    }

    float y, z;

    if (x <= 2.0) {
        y = x * x - 2.0;
        y = chbevl(y, k0_AF, 10) - logf(0.5 * x) * cyl_bessel_i0f(x);
        return y;
    }

    z = 8.0 / x - 2.0;
    y = chbevl(z, k0_BF, 25) / sqrtf(x);
    return y * expf(-x);
}

__device__ double k1(double x){
    if (x == 0) {
        return CUDART_INF;
    }

    if (x < 0) {
        return CUDART_NAN;
    }

    double y;

    if (x <= 2.0) {
        y = x * x - 2.0;
        y = log(0.5 * x) * cyl_bessel_i1(x) + chbevl(y, k1_A, 11) / x;
        return y;
    }

    return (exp(-x) * chbevl(8.0 / x - 2.0, k1_B, 25) / sqrt(x));
}

__device__ float k1f(float x){
    if (x == 0) {
        return CUDART_INF;
    }

    if (x < 0) {
        return CUDART_NAN;
    }

    float y;
    float i1;

    if (x <= 2.0) {
        y = x * x - 2.0;
        i1 = cyl_bessel_i1f(x);
        y = logf(0.5 * x) * i1 + chbevl(y, k1_AF, 11) / x;
        return y;
    }

    float z = 8.0 / x - 2.0;
    return (expf(-x) * chbevl(z, k1_BF, 25) / sqrtf(x));
}
"""


k0 = _core.create_ufunc(
    'cupyx_scipy_special_k0',
    (('f->f', 'out0 = k0f(in0)'), 'd->d'),
    'out0 = k0(in0)',
    preamble=k_preamble,
    doc='''Modified Bessel function of the second kind of order 0.

    Args:
        x (cupy.ndarray): argument (float)

    Returns:
        cupy.ndarray: Value of the modified Bessel function K of order 0 at x.

    .. seealso:: :meth:`scipy.special.k0`

    ''')


k0e = _core.create_ufunc(
    'cupyx_scipy_special_k0e',
    (('f->f', 'out0 = expf(in0) * k0f(in0)'), 'd->d'),
    'out0 = exp(in0) * k0(in0)',
    preamble=k_preamble,
    doc='''Exponentially scaled modified Bessel function K of order 0

    Args:
        x (cupy.ndarray): argument (float)

    Returns:
        cupy.ndarray: Value at x.

    .. seealso:: :meth:`scipy.special.k0e`

    ''')


k1 = _core.create_ufunc(
    'cupyx_scipy_special_k1',
    (('f->f', 'out0 = k1f(in0)'), 'd->d'),
    'out0 = k1(in0)',
    preamble=k_preamble,
    doc='''Modified Bessel function of the second kind of order 1.

    Args:
        x (cupy.ndarray): argument (float)

    Returns:
        cupy.ndarray: Value of the modified Bessel function K of order 1 at x.

    .. seealso:: :meth:`scipy.special.k1`

    ''')


k1e = _core.create_ufunc(
    'cupyx_scipy_special_k1e',
    (('f->f', 'out0 = expf(in0) * k1f(in0)'), 'd->d'),
    'out0 = exp(in0) * k1(in0)',
    preamble=k_preamble,
    doc='''Exponentially scaled modified Bessel function K of order 1

    Args:
        x (cupy.ndarray): argument (float)

    Returns:
        cupy.ndarray: Value at x.

    .. seealso:: :meth:`scipy.special.k1e`

    ''')
