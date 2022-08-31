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


k_implementation = """
#include <cupy/math_constants.h>

/* Chebyshev coefficients for K0(x) + log(x/2) I0(x)
 * in the interval [0,2].  The odd order coefficients are all
 * zero; only the even order coefficients are listed.
 *
 * lim(x->0){ K0(x) + log(x/2) I0(x) } = -EUL.
 */
__device__ double A[] = {
    1.37446543561352307156E-16,
    0,
    4.25981614279661018399E-14,
    0,
    1.03496952576338420167E-11,
    0,
    1.90451637722020886025E-9,
    0,
    2.53479107902614945675E-7,
    0,
    2.28621210311945178607E-5,
    0,
    1.26461541144692592338E-3,
    0,
    3.59799365153615016266E-2,
    0,
    3.44289899924628486886E-1,
    0,
    -5.35327393233902768720E-1
};

/* Chebyshev coefficients for exp(x) sqrt(x) K0(x)
 * in the inverted interval [2,infinity].
 *
 * lim(x->inf){ exp(x) sqrt(x) K0(x) } = sqrt(pi/2).
 */
__device__ double B[] = {
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

// template<typename T>
// __device__ T k0e(T x){
__device__ double k0(double x){
    if (x == 0) {
        return CUDART_INF;
    }
    
    if (x < 0) {
        return CUDART_NAN;
    }

    if (x <= 2.0) {
        double y = x * x - 2.0;
        y = chbevl(y, A, 19) - log(0.5 * x) * cyl_bessel_i0(x);
        return (y);
    }

    return chbevl(8.0 / x - 2.0, B, 25) / sqrt(x);
}

"""

k0 = _core.create_ufunc(
    'cupyx_scipy_special_k0e',
    ('f->f', 'd->d'),
    'out0 = exp(-in0)*k0(in0)',
    preamble=chbevl_implementation + k_implementation,
    doc='''Modified Bessel function of order 1.

    .. seealso:: :meth:`scipy.special.i1`

    ''')

k0e = _core.create_ufunc(
    'cupyx_scipy_special_k0',
    ('f->f', 'd->d'),
    'out0 = k0(in0)',
    preamble=chbevl_implementation+k_implementation,
    doc='''Modified Bessel function of order 1.

    .. seealso:: :meth:`scipy.special.i1`

    ''')
