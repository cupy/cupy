from cupy.math import ufunc


sinh = ufunc.create_math_ufunc(
    'sinh', 1, 'cupy_sinh',
    '''Elementwise hyperbolic sine function.

    .. seealso:: :data:`numpy.sinh`

    ''')


cosh = ufunc.create_math_ufunc(
    'cosh', 1, 'cupy_cosh',
    '''Elementwise hyperbolic cosine function.

    .. seealso:: :data:`numpy.cosh`

    ''')


tanh = ufunc.create_math_ufunc(
    'tanh', 1, 'cupy_tanh',
    '''Elementwise hyperbolic tangent function.

    .. seealso:: :data:`numpy.tanh`

    ''')


arcsinh = ufunc.create_math_ufunc(
    'asinh', 1, 'cupy_arcsinh',
    '''Elementwise inverse of hyperbolic sine function.

    .. seealso:: :data:`numpy.arcsinh`

    ''')


arccosh = ufunc.create_math_ufunc(
    'acosh', 1, 'cupy_arccosh',
    '''Elementwise inverse of hyperbolic cosine function.

    .. seealso:: :data:`numpy.arccosh`

    ''')


arctanh = ufunc.create_math_ufunc(
    'atanh', 1, 'cupy_arctanh',
    '''Elementwise inverse of hyperbolic tangent function.

    .. seealso:: :data:`numpy.arctanh`

    ''')
