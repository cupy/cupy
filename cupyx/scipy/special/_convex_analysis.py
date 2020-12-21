from cupy import core


_float_preamble = '''
#include <cupy/math_constants.h>

double __device__ entr(double x) {
    if (isnan(x)) {
        return CUDART_NAN;
    } else if (x > 0){
        return -x * log(x);
    } else if (x == 0){
        return 0;
    } else {
        return -CUDART_INF;
    }
}

double __device__ kl_div(double x, double y) {
    if (isnan(x) || isnan(y)) {
        return CUDART_NAN;
    } else if (x > 0 && y > 0) {
        return x * log(x / y) - x + y;
    } else if (x == 0 && y >= 0) {
        return y;
    } else {
        return CUDART_INF;
    }
}

double __device__ rel_entr(double x, double y) {
    if (isnan(x) || isnan(y)) {
        return CUDART_NAN;
    } else if (x > 0 && y > 0) {
        return x * log(x / y);
    } else if (x == 0 && y >= 0) {
        return 0;
    } else {
        return CUDART_INF;
    }
}

double __device__ huber(double delta, double r) {
    if (delta < 0) {
        return CUDART_INF;
    } else if (abs(r) <= delta) {
        return 0.5 * r * r;
    } else {
        return delta * (abs(r) - 0.5 * delta);
    }
}

double __device__ pseudo_huber(double delta, double r) {
    if (delta < 0) {
        return CUDART_INF;
    } else if (delta == 0 || r == 0) {
        return 0;
    } else {
        double u = delta;
        double v = r / delta;
        return u * u * (sqrt(1 + v * v) - 1);
    }
}

'''


entr = core.create_ufunc(
    'cupyx_scipy_entr', ('f->f', 'd->d'),
    'out0 = out0_type(entr(in0));',
    preamble=_float_preamble,
    doc='''Elementwise function for computing entropy.

    .. seealso:: :meth:`scipy.special.entr`

    ''')


kl_div = core.create_ufunc(
    'cupyx_scipy_kl_div', ('ff->f', 'dd->d'),
    'out0 = out0_type(kl_div(in0, in1));',
    preamble=_float_preamble,
    doc='''Elementwise function for computing Kullback-Leibler divergence.

    .. seealso:: :meth:`scipy.special.kl_div`

    ''')


rel_entr = core.create_ufunc(
    'cupyx_scipy_rel_entr', ('ff->f', 'dd->d'),
    'out0 = out0_type(rel_entr(in0, in1));',
    preamble=_float_preamble,
    doc='''Elementwise function for computing relative entropy.

    .. seealso:: :meth:`scipy.special.rel_entr`

    ''')


huber = core.create_ufunc(
    'cupyx_scipy_huber', ('ff->f', 'dd->d'),
    'out0 = out0_type(huber(in0, in1));',
    preamble=_float_preamble,
    doc='''Elementwise function for computing the Huber loss.

    .. seealso:: :meth:`scipy.special.huber`

    ''')


pseudo_huber = core.create_ufunc(
    'cupyx_scipy_pseudo_huber', ('ff->f', 'dd->d'),
    'out0 = out0_type(pseudo_huber(in0, in1));',
    preamble=_float_preamble,
    doc='''Elementwise function for computing the Pseudo-Huber loss.

    .. seealso:: :meth:`scipy.special.pseudo_huber`

    ''')
