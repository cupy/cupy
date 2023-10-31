import numpy
import cupy
import cupy as xp


x = cupy.array([1, 2, 3])

xp.empty(10)
xp.empty((10, 20))
xp.empty(numpy.array([10, 20]))
cupy.empty()  # E: Missing positional argument "shape" in call to "empty"  [call-arg]
cupy.empty(10, 20)  # E: Argument 2 to "empty" has incompatible type "int"; [arg-type]
# xp.empty(cupy.array([10, 20]))  # TODO(asi1024): Fix to fail typecheck
xp.empty(x.shape)
xp.empty((10, 20), float)
xp.empty((10, 20), int)
xp.empty((10, 20), numpy.float32)
xp.empty((10, 20), 'i4')
xp.empty((10, 20), 'int32')
numpy.empty((10, 20), numpy.datetime64)
# cupy.empty((10, 20), numpy.datetime64)  # TODO(asi1024): Fix to fail typecheck
xp.empty((10, 20), float, 'C')
xp.empty((10, 20), float, 'F')
xp.empty((10, 20), float, 'K')  # E: Argument 3 to "empty" has incompatible type [arg-type]
xp.empty((10, 20), float, 'A')  # E: Argument 3 to "empty" has incompatible type [arg-type]
xp.empty((10, 20), float, 'X')  # E: Argument 3 to "empty" has incompatible type [arg-type]
xp.empty(shape=(10, 20), dtype=float, order='C')

xp.empty_like(x)
xp.empty_like(x, float)
xp.empty_like(x, float, 'C')
xp.empty_like(x, float, 'F')
xp.empty_like(x, float, 'K')
xp.empty_like(x, float, 'A')
xp.empty_like(prototype=x, dtype=float, order='C', shape=(10, 20))

xp.eye(10)
xp.eye(10, 20)
xp.eye(10, 20, 3)
xp.eye(10, 20, 3, float, 'C')
xp.eye(10, 20, 3, float, 'F')
xp.eye(10, 20, 3, float, 'K')  # E: Argument 5 to "eye" has incompatible type [arg-type]
xp.eye(10, 20, 3, float, 'A')  # E: Argument 5 to "eye" has incompatible type [arg-type]
xp.eye(N=10, M=20, k=3, dtype=float, order='C')

xp.identity(10)
xp.identity(10, float)
xp.identity(n=10, dtype=float)

xp.ones((10, 20))
xp.ones((10, 20), float)
xp.ones((10, 20), float, 'C')
xp.ones((10, 20), float, 'F')
xp.ones((10, 20), float, 'K')  # E: Argument 3 to "ones" has incompatible type [arg-type]
xp.ones((10, 20), float, 'A')  # E: Argument 3 to "ones" has incompatible type [arg-type]

xp.ones_like(x)
xp.ones_like(x, float)
xp.ones_like(x, float, 'C')
xp.ones_like(x, float, 'F')
xp.ones_like(x, float, 'K')
xp.ones_like(x, float, 'A')
xp.ones_like(a=x, dtype=float, order='C', shape=(10, 20))

xp.zeros((10, 20))
xp.zeros((10, 20), float)
xp.zeros((10, 20), float, 'C')
xp.zeros((10, 20), float, 'F')
xp.zeros((10, 20), float, 'K')  # E: Argument 3 to "zeros" has incompatible type [arg-type]
xp.zeros((10, 20), float, 'A')  # E: Argument 3 to "zeros" has incompatible type [arg-type]

xp.zeros_like(x)
xp.zeros_like(x, float)
xp.zeros_like(x, float, 'C')
xp.zeros_like(x, float, 'F')
xp.zeros_like(x, float, 'K')
xp.zeros_like(x, float, 'A')
xp.zeros_like(a=x, dtype=float, order='C', shape=(10, 20))

xp.full((10, 20))  # E: Missing positional argument "fill_value" [call-arg]
# xp.full((10, 20), "abc")  # TODO(asi1024): Fix to fail typecheck
xp.full((10, 20), 30)
xp.full((10, 20), 30, float)
xp.full((10, 20), 30, float, 'C')
xp.full((10, 20), 30, float, 'F')
xp.full((10, 20), 30, float, 'K')  # E: Argument 4 to "full" has incompatible type [arg-type]
xp.full((10, 20), 30, float, 'A')  # E: Argument 4 to "full" has incompatible type [arg-type]

xp.full_like(x, 30)
xp.full_like(x, 30, float)
xp.full_like(x, 30, float, 'C')
xp.full_like(x, 30, float, 'F')
xp.full_like(x, 30, float, 'K')
xp.full_like(x, 30, float, 'A')
xp.full_like(a=x, fill_value=30, dtype=float, order='C', shape=(10, 20))
