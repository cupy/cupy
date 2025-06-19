from cupy import _core


# Note: complex-valued isnan, log and log1p are all defined in
#       cupy/_core/include/cupy/complex.cuh
xlogy_definition = """

template <typename T>
static __device__ T xlogy(T x, T y) {
    if ((x == (T)0.0) && !isnan(y)) {
        return (T)0.0;
    } else {
        return x * log(y);
    }
}

"""


# Note: SciPy only defines dd->d and DD->D
xlogy = _core.create_ufunc(
    'cupy_xlogy',
    ('ee->f', 'ff->f', 'dd->d', 'FF->F', 'DD->D'),
    'out0 = out0_type(xlogy(in0, in1));',
    preamble=xlogy_definition,
    doc='''Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.

    Args:
        x (cupy.ndarray): input data

    Returns:
        cupy.ndarray: values of ``x * log(y)``

    .. seealso:: :data:`scipy.special.xlogy`

    ''')


xlog1py_definition = """

template <typename T>
static __device__ T xlog1py(T x, T y) {
    if ((x == (T)0.0) && ~isnan(y)) {
        return (T)0.0;
    } else {
        return x * log1p(y);
    }
}

"""

# Note: SciPy only defines dd->d and DD->D
xlog1py = _core.create_ufunc(
    'cupy_xlog1py',
    ('ee->f', 'ff->f', 'dd->d', 'FF->F', 'DD->D'),
    'out0 = out0_type(xlog1py(in0, in1));',
    preamble=xlog1py_definition,
    doc='''Compute ``x*log1p(y)`` so that the result is 0 if ``x = 0``.

    Args:
        x (cupy.ndarray): input data

    Returns:
        cupy.ndarray: values of ``x * log1p(y)``

    .. seealso:: :data:`scipy.special.xlog1py`

    ''')
