import numpy
import cupy
# cupy.einsum("ii", cupy.array([[0, 1],[3, 4]]))
# cupy.array([[0, 1],[3, 4]]).sum()
print(numpy.einsum("ii", numpy.array([[2, 1],[3, 4]])))
print(cupy.einsum("ii", cupy.array([[2, 1],[3, 4]])))
# print(numpy.einsum("...", 0))


