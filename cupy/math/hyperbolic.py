from cupy.math import ufunc

sinh = ufunc.create_math_ufunc('sinh', 1)
cosh = ufunc.create_math_ufunc('cosh', 1)
tanh = ufunc.create_math_ufunc('tanh', 1)
arcsinh = ufunc.create_math_ufunc('asinh', 1, 'arcsinh')
arccosh = ufunc.create_math_ufunc('acosh', 1, 'arccosh')
arctanh = ufunc.create_math_ufunc('atanh', 1, 'arctanh')
