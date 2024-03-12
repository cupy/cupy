# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import cupy

_cupy_kernel_cache = {}

_SUPPORTED_TYPES = ["float32", "float64"]

_cuda_code_kalman = """
// Compute linalg.inv(S)
template<typename T, int BLOCKS, int DIM_Z>
__device__ T inverse(
    const int & ltx,
    const int & lty,
    const int & ltz,
    T(&s_ZZ_A)[BLOCKS][DIM_Z][DIM_Z],
    T(&s_ZZ_I)[BLOCKS][DIM_Z][DIM_Z]) {

    T temp {};

    // Interchange the row of matrix
    if ( lty == 0 && ltx < DIM_Z) {
#pragma unroll ( DIM_Z - 1 )
        for ( int i = DIM_Z - 1; i > 0; i-- ) {
            if ( s_ZZ_A[ltz][i - 1][0] < s_ZZ_A[ltz][i][0] ) {
                    temp = s_ZZ_A[ltz][i][ltx];
                    s_ZZ_A[ltz][i][ltx] = s_ZZ_A[ltz][i - 1][ltx];
                    s_ZZ_A[ltz][i - 1][ltx] = temp;

                    temp = s_ZZ_I[ltz][i][ltx];
                    s_ZZ_I[ltz][i][ltx] = s_ZZ_I[ltz][i - 1][ltx];
                    s_ZZ_I[ltz][i - 1][ltx] = temp;
            }
        }
    }

    // Replace a row by sum of itself and a
    // constant multiple of another row of the matrix
#pragma unroll DIM_Z
    for ( int i = 0; i < DIM_Z; i++ ) {
        if ( lty < DIM_Z && ltx < DIM_Z ) {
            if ( lty != i ) {
                temp = s_ZZ_A[ltz][lty][i] / s_ZZ_A[ltz][i][i];
            }
        }

        __syncthreads();
        if ( lty < DIM_Z && ltx < DIM_Z ) {
            if ( lty != i ) {
                s_ZZ_A[ltz][lty][ltx] -= s_ZZ_A[ltz][i][ltx] * temp;
                s_ZZ_I[ltz][lty][ltx] -= s_ZZ_I[ltz][i][ltx] * temp;
            }
        }
        __syncthreads();
    }

    if ( lty < DIM_Z && ltx < DIM_Z ) {
        // Multiply each row by a nonzero integer.
        // Divide row element by the diagonal element
        temp = s_ZZ_A[ltz][lty][lty];
    }
    __syncthreads();

    if ( lty < DIM_Z && ltx < DIM_Z ) {
        s_ZZ_A[ltz][lty][ltx] = s_ZZ_A[ltz][lty][ltx] / temp;
        s_ZZ_I[ltz][lty][ltx] = s_ZZ_I[ltz][lty][ltx] / temp;
    }

    __syncthreads();

    return ( s_ZZ_I[ltz][lty][ltx] );
}


template<typename T, int BLOCKS, int DIM_X, int DIM_U, int MAX_TPB>
__global__ void __launch_bounds__(MAX_TPB) _cupy_predict(
        const int num_points,
        const T * __restrict__ alpha_sq,
        T * __restrict__ x_in,
        const T * __restrict__ u,
        const T * __restrict__ B,
        const T * __restrict__ F,
        T * __restrict__ P,
        const T * __restrict__ Q,
        const bool skip
        ) {

    __shared__ T s_XX_A[BLOCKS][DIM_X][DIM_X];
    __shared__ T s_XX_F[BLOCKS][DIM_X][DIM_X];
    __shared__ T s_XX_P[BLOCKS][DIM_X][DIM_X];

    const auto ltx = threadIdx.x;
    const auto lty = threadIdx.y;
    const auto ltz = threadIdx.z;

    const int btz { static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) };

    const int stride_z { static_cast<int>( blockDim.z * gridDim.z ) };

    const int x_value { lty * DIM_X + ltx };

    for ( int gtz = btz; gtz < num_points; gtz += stride_z ) {

        s_XX_F[ltz][lty][ltx] = F[gtz * DIM_X * DIM_X + x_value];

        __syncthreads();

        T alpha2 { alpha_sq[gtz] };
        T localQ { Q[gtz * DIM_X * DIM_X + x_value] };
        T localP { P[gtz * DIM_X * DIM_X + x_value] };

        T temp {};
        //T temp2 {};

        /*
        if ( !skip ) {
            // Compute self.x = dot(B, u)
            if ( ltx == 0 ) {
#pragma unroll DIM_U
                for ( int j = 0; j < DIM_U; j++ ) {
                    temp2 += B[gtz * DIM_X * DIM_U + lty * DIM_U + j] *
                        u[gtz * DIM_U + j];
                }
                printf("%d: %f\\n", lty, temp2);
            }
        }
        */

        // Compute self.x = dot(F, self.x)
        if ( ltx == 0 ) {
#pragma unroll DIM_X
            for ( int j = 0; j < DIM_X; j++ ) {
                temp += s_XX_F[ltz][lty][j] *
                    x_in[gtz * DIM_X + j + ltx];
            }
            // x_in[gtz * DIM_X * 1 + lty * 1 + ltx]
            //x_in[gtz * DIM_X + lty + ltx] = temp + temp2;
            x_in[gtz * DIM_X + lty + ltx] = temp;
        }

        s_XX_P[ltz][lty][ltx] = localP;

        __syncthreads();

        // Compute dot(F, self.P)
        temp = 0.0;
#pragma unroll DIM_X
        for ( int j = 0; j < DIM_X; j++ ) {
            temp += s_XX_F[ltz][lty][j] *
                s_XX_P[ltz][j][ltx];
        }
        s_XX_A[ltz][lty][ltx] = temp;

        __syncthreads();

        // Compute dot(dot(F, self.P), F.T)
        temp = 0.0;
#pragma unroll DIM_X
        for ( int j = 0; j < DIM_X; j++ ) {
            temp += s_XX_A[ltz][lty][j] *   //133
                s_XX_F[ltz][ltx][j];
        }

        __syncthreads();

        // Compute self._alpha_sq * dot(dot(F, self.P), F.T) + Q
        // Where temp = dot(dot(F, self.P), F.T)
        P[gtz * DIM_X * DIM_X + x_value] =
            alpha2 * temp + localQ;
    }
}


template<typename T, int BLOCKS, int DIM_X, int DIM_Z, int MAX_TPB>
__global__ void __launch_bounds__(MAX_TPB) _cupy_update(
        const int num_points,
        T * __restrict__ x_in,
        const T * __restrict__ z_in,
        const T * __restrict__ H,
        T * __restrict__ P,
        const T * __restrict__ R
        ) {

    __shared__ T s_XX_A[BLOCKS][DIM_X][DIM_X];
    __shared__ T s_XX_B[BLOCKS][DIM_X][DIM_X];
    __shared__ T s_XX_P[BLOCKS][DIM_X][DIM_X];
    __shared__ T s_ZX_H[BLOCKS][DIM_Z][DIM_X];
    __shared__ T s_XZ_K[BLOCKS][DIM_X][DIM_Z];
    __shared__ T s_XZ_A[BLOCKS][DIM_X][DIM_Z];
    __shared__ T s_ZZ_A[BLOCKS][DIM_Z][DIM_Z];
    __shared__ T s_ZZ_R[BLOCKS][DIM_Z][DIM_Z];
    __shared__ T s_ZZ_I[BLOCKS][DIM_Z][DIM_Z];
    __shared__ T s_Z1_y[BLOCKS][DIM_Z][1];

    const auto ltx = threadIdx.x;
    const auto lty = threadIdx.y;
    const auto ltz = threadIdx.z;

    const int btz {
        static_cast<int>( blockIdx.z * blockDim.z + threadIdx.z ) };

    const int stride_z { static_cast<int>( blockDim.z * gridDim.z ) };

    const int x_value { lty * DIM_X + ltx };
    const int z_value { lty * DIM_Z + ltx };

    for ( int gtz = btz; gtz < num_points; gtz += stride_z ) {

        if ( lty < DIM_Z ) {
            s_ZX_H[ltz][lty][ltx] =
                H[gtz * DIM_Z * DIM_X + x_value];
        }

        __syncthreads();

        s_XX_P[ltz][lty][ltx] = P[gtz * DIM_X * DIM_X + x_value];

        if ( ( lty < DIM_Z ) && ( ltx < DIM_Z ) ) {
            s_ZZ_R[ltz][lty][ltx] =
                R[gtz * DIM_Z * DIM_Z + z_value];

            if ( lty == ltx ) {
                s_ZZ_I[ltz][lty][ltx] = 1.0;
            } else {
                s_ZZ_I[ltz][lty][ltx] = 0.0;
            }
        }

        T temp {};

        // Compute self.y : z = dot(self.H, self.x) --> Z1
        if ( ( ltx == 0 ) && ( lty < DIM_Z ) ) {
            T temp_z { z_in[gtz * DIM_Z + lty] };

#pragma unroll DIM_X
            for ( int j = 0; j < DIM_X; j++ ) {
                temp += s_ZX_H[ltz][lty][j] *
                    x_in[gtz * DIM_X + j];
            }

            s_Z1_y[ltz][lty][ltx] = temp_z - temp;
        }

        __syncthreads();

        // Compute PHT : dot(self.P, self.H.T) --> XZ
        temp = 0.0;
        if ( ltx < DIM_Z ) {
#pragma unroll DIM_X
            for ( int j = 0; j < DIM_X; j++ ) {
                temp += s_XX_P[ltz][lty][j] *
                    s_ZX_H[ltz][ltx][j];
            }
            // s_XX_A holds PHT
            s_XZ_A[ltz][lty][ltx] = temp;
        }

        __syncthreads();

        // Compute self.S : dot(self.H, PHT) + self.R --> ZZ
        temp = 0.0;
        if ( ( ltx < DIM_Z ) && ( lty < DIM_Z ) ) {
#pragma unroll DIM_X
            for ( int j = 0; j < DIM_X; j++ ) {
                temp += s_ZX_H[ltz][lty][j] *
                    s_XZ_A[ltz][j][ltx];
            }
            // s_XX_B holds S - system uncertainty
            s_ZZ_A[ltz][lty][ltx] = temp + s_ZZ_R[ltz][lty][ltx];
        }

        __syncthreads();

        // Compute matrix inversion
        temp = inverse(ltx, lty, ltz, s_ZZ_A, s_ZZ_I);

        __syncthreads();

        if ( ( ltx < DIM_Z ) && ( lty < DIM_Z ) ) {
            // s_XX_B hold SI - inverse system uncertainty
            s_ZZ_A[ltz][lty][ltx] = temp;
        }

        __syncthreads();

        //  Compute self.K : dot(PHT, self.SI) --> ZZ
        //  kalman gain
        temp = 0.0;
        if ( ltx < DIM_Z ) {
#pragma unroll DIM_Z
            for ( int j = 0; j < DIM_Z; j++ ) {
                temp += s_XZ_A[ltz][lty][j] *
                    s_ZZ_A[ltz][ltx][j];
            }
            s_XZ_K[ltz][lty][ltx] = temp;
        }

        __syncthreads();

        //  Compute self.x : self.x + cp.dot(self.K, self.y) --> X1
        temp = 0.0;
        if ( ltx == 0 ) {
#pragma unroll DIM_Z
            for ( int j = 0; j < DIM_Z; j++ ) {
                temp += s_XZ_K[ltz][lty][j] *
                s_Z1_y[ltz][j][ltx];
            }
            x_in[gtz * DIM_X * 1 + lty * 1 + ltx] += temp;
        }

        // Compute I_KH = self_I - dot(self.K, self.H) --> XX
        temp = 0.0;
#pragma unroll DIM_Z
        for ( int j = 0; j < DIM_Z; j++ ) {
            temp += s_XZ_K[ltz][lty][j] *
                s_ZX_H[ltz][j][ltx];
        }
        // s_XX_A holds I_KH
        s_XX_A[ltz][lty][ltx] = ( ( ltx == lty ) ? 1 : 0 ) - temp;

        __syncthreads();

        // Compute self.P = dot(dot(I_KH, self.P), I_KH.T) +
        // dot(dot(self.K, self.R), self.K.T)

        // Compute dot(I_KH, self.P) --> XX
        temp = 0.0;
#pragma unroll DIM_X
        for ( int j = 0; j < DIM_X; j++ ) {
            temp += s_XX_A[ltz][lty][j] *
                s_XX_P[ltz][j][ltx];
        }
        s_XX_B[ltz][lty][ltx] = temp;

        __syncthreads();

        // Compute dot(dot(I_KH, self.P), I_KH.T) --> XX
        temp = 0.0;
#pragma unroll DIM_X
        for ( int j = 0; j < DIM_X; j++ ) {
            temp += s_XX_B[ltz][lty][j] *
                s_XX_A[ltz][ltx][j];
        }

        s_XX_P[ltz][lty][ltx] = temp;

        // Compute dot(self.K, self.R) --> XZ
        temp = 0.0;
        if ( ltx < DIM_Z ) {
#pragma unroll DIM_Z
            for ( int j = 0; j < DIM_Z; j++ ) {
                temp += s_XZ_K[ltz][lty][j] *
                    s_ZZ_R[ltz][j][ltx];
            }

            // s_XZ_A holds dot(self.K, self.R)
            s_XZ_A[ltz][lty][ltx] = temp;
        }

        __syncthreads();

        // Compute dot(dot(self.K, self.R), self.K.T) --> XX
        temp = 0.0;
#pragma unroll DIM_Z
        for ( int j = 0; j < DIM_Z; j++ ) {
            temp += s_XZ_A[ltz][lty][j] *
                s_XZ_K[ltz][ltx][j];
        }

        P[gtz * DIM_X * DIM_X + x_value] =
            s_XX_P[ltz][lty][ltx] + temp;
    }
}
"""


class _cupy_predict_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(
        self,
        alpha_sq,
        x,
        u,
        B,
        F,
        P,
        Q,
    ):

        if B is not None and u is not None:
            skip = False
        else:
            skip = True

        kernel_args = (x.shape[0], alpha_sq, x, u, B, F, P, Q, skip)

        self.kernel(self.grid, self.block, kernel_args)


class _cupy_update_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(self, x, z, H, P, R):

        kernel_args = (x.shape[0], x, z, H, P, R)

        self.kernel(self.grid, self.block, kernel_args)


def _populate_kernel_cache(np_type, blocks, dim_x, dim_z, dim_u, max_tpb):

    # Check in np_type is a supported option
    if np_type not in _SUPPORTED_TYPES:
        raise ValueError(
            "Datatype {} not found for Kalman Filter".format(np_type))

    if np_type == "float32":
        c_type = "float"
    else:
        c_type = "double"

    #     Check CuPy version
    # Update to only check for v8.X in cuSignal 0.16
    # Instantiate the cupy kernel for this type and compile
    specializations = (
        "_cupy_predict<{}, {}, {}, {}, {}>".format(
            c_type, blocks, dim_x, dim_u, max_tpb
        ),
        "_cupy_update<{}, {}, {}, {}, {}>".format(
            c_type, blocks, dim_x, dim_z, max_tpb
        ),
    )
    module = cupy.RawModule(
        code=_cuda_code_kalman,
        options=(
            "-std=c++11",
            "-fmad=true",
        ),
        name_expressions=specializations,
    )

    _cupy_kernel_cache[(str(np_type), "predict")] = module.get_function(
        specializations[0]
    )

    _cupy_kernel_cache[(str(np_type), "update")] = module.get_function(
        specializations[1]
    )


def _get_backend_kernel(dtype, grid, block, k_type):

    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
    if kernel:
        if k_type == "predict":
            return _cupy_predict_wrapper(grid, block, kernel)
        elif k_type == "update":
            return _cupy_update_wrapper(grid, block, kernel)
        else:
            raise NotImplementedError(
                "No CuPY kernel found for k_type {}, datatype {}".format(
                    k_type, dtype)
            )
    else:
        raise ValueError(
            "Kernel {} not found in _cupy_kernel_cache".format(k_type))
