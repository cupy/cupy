import sys
import numpy
import cupy

code = '''
__device__ double3 operator+(const double3& lhs, const double3& rhs) {
    return make_double3(lhs.x + rhs.x,
                        lhs.y + rhs.y,
                        lhs.z + rhs.z);
}

extern "C" __global__ void sum_kernel(const double3* lhs,
                                            double3  rhs,
                                            double3* out) {
  int i = threadIdx.x;
  out[i] = lhs[i] + rhs;
}
'''

double3 = numpy.dtype(
    {
        'names': ['x', 'y', 'z'],
        'formats': [numpy.float64]*3
    }
)


def main():
    N = 8

    # The kernel computes out = lhs+rhs where lhs and rhs are double3 vectors.
    # lhs is an array of N such vectors and rhs is double3 kernel parameter.

    lhs = cupy.random.rand(3*N, dtype=numpy.float64).reshape(N, 3)
    rhs = numpy.random.rand(3).astype(numpy.float64)
    out = cupy.empty_like(lhs)

    kernel = cupy.RawKernel(code, 'sum_kernel')
    args = (lhs, rhs.view(double3), out)
    kernel((1,), (N,), args)

    expected = lhs + cupy.asarray(rhs[None, :])
    cupy.testing.assert_array_equal(expected, out)
    print("Kernel output matches expected value.")


if __name__ == '__main__':
    sys.exit(main())
