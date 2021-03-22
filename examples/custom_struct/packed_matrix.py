import sys
import numpy
import cupy

code = '''
template<typename T>
struct Matrix {
    T value[4][4];

    __device__ T& operator() (int i, int j) {
        return this->value[i][j];
    }

    __device__ const T& operator() (int i, int j) const {
        return this->value[i][j];
    }
};

template<typename T>
__device__ Matrix<T> operator+ (const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res;
    for (int i = 0; i<4; i++) {
        for (int j = 0; j<4; j++) {
            res(i,j) = lhs(i,j) + rhs(i,j);
        }
    }
    return res;
}

template<typename T>
__device__ Matrix<T> operator* (const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res;
    for (int i = 0; i<4; i++) {
        for (int j = 0; j<4; j++) {
            res(i,j) = T(0);
            for (int k = 0; k<4; k++) {
                res(i,j) += lhs(i,k) * rhs(k,j);
            }
        }
    }
    return res;
}

template<typename T>
__global__ void kernel(const Matrix<T>* A,
                       const Matrix<T>* B,
                       const Matrix<T>  C,
                             Matrix<T>* out) {
  int i = threadIdx.x;
  out[i] = A[i] * B[i] + C;
}
'''


def main():
    N = 8
    module = cupy.RawModule(code=code, options=('-std=c++11',),
                            name_expressions=('kernel<float>',
                                              'kernel<double>'))

    # The kernel computes out = A*B+C where A, B and C are 4x4 matrices.
    # A and B are arrays of N such matrices and C is a matrix kernel parameter.

    for (ctype, dtype) in zip(('float', 'double'),
                              (numpy.float32, numpy.float64)):

        A = cupy.random.rand(16*N, dtype=dtype).reshape(N, 4, 4)
        B = cupy.random.rand(16*N, dtype=dtype).reshape(N, 4, 4)
        C = numpy.random.rand(16).astype(dtype).reshape(4, 4)
        out = cupy.empty_like(A)

        Matrix = numpy.dtype(
            {
                'names': ['value'],
                'formats': [(dtype, (4, 4))]
            }
        )

        kernel = module.get_function('kernel<{}>'.format(ctype))
        args = (A, B, C.ravel().view(Matrix), out)
        kernel((1,), (N,), args)

        expected = cupy.matmul(A, B) + cupy.asarray(C[None, :, :])

        cupy.testing.assert_array_almost_equal(expected, out)
        print("Kernel output matches expected value for "
              "type '{}'.".format(ctype))


if __name__ == '__main__':
    sys.exit(main())
