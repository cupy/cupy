#include <stdint.h>
#include <stdio.h>

//#define DIM_X  ${DIM_X}
//#define DIM_Y  ${DIM_Y}
//#define BLK_M  ${BLK_M}
//#define BLK_N  ${BLK_N}
//#define BLK_K  ${BLK_K}
//#define DIM_XA  ${DIM_XA}
//#define DIM_YA  ${DIM_YA}
//#define DIM_XB  ${DIM_XB}
//#define DIM_YB  ${DIM_YB}
//#define THR_N  ${THR_N}
//#define THR_M  ${THR_M}
//

// LDA is M
// LDB is K


extern "C" __global__
void sgemm(
        int M, int N, int K,
        const float* A, int LDA,
        const float* B, int LDB,
        float * C, int LDC)
{
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int idt = DIM_X * idy + idx;

    int idxA = idt % DIM_XA;
    int idyA = idt / DIM_XA;

    int idxB = idt % DIM_XB;
    int idyB = idt / DIM_XB;

    int blx = blockIdx.x;
    int bly = blockIdx.y;

    __shared__ float sA[BLK_K][BLK_M + 1];
    __shared__ float sB[BLK_N][BLK_K + 1];

    // registers for the innermost loop
    // performance was highest when type of rC was fixed to uint32
    // Note that the performance was highest when C's type was short
    float rC[THR_N][THR_M];
    float rA[THR_M];
    float rB[THR_N];

    float ra[BLK_K / DIM_YA][BLK_M / DIM_XA];
    float rb[BLK_N / DIM_YB][BLK_K / DIM_XB];

    const float* offs_dA = A + blx * BLK_M       + idyA * LDA + idxA;
    int boundA = (LDA * (K - 1) + M) - (blx * BLK_M + idyA * LDA + idxA) - 1;
    const float* offs_dB = B + bly * BLK_N * LDB + idyB * LDB + idxB;
    int boundB = (LDB * (N - 1) + K) - (bly * BLK_N * LDB + idyB * LDB + idxB) - 1;

    int m, n, k, kk;
    
    #pragma unroll
    for (n = 0; n < THR_N; n++)
        #pragma unroll
        for (m = 0 ; m < THR_M; m++)
            rC[n][m] = 0;

    // blockwise transpose to transpose load
    #pragma unroll
    for (n = 0; n < BLK_K; n += DIM_YA)
        #pragma unroll
        for (m = 0; m < BLK_M; m += DIM_XA)
            sA[n + idyA][m + idxA] = offs_dA[n * LDA + m];
    // blockwise transpose to transpose load
    #pragma unroll
    for (n = 0; n < BLK_N; n += DIM_YB)
        #pragma unroll
        for (m = 0; m < BLK_K; m += DIM_XB)
            sB[n + idyB][m + idxB] = offs_dB[n * LDB + m];
    __syncthreads();

    for (kk = 0; kk < K - BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K * LDA;
        boundA -= BLK_K * LDA;
        offs_dB += BLK_K;
        boundB -= BLK_K;
        
        #pragma unroll
        for (n = 0; n < BLK_K / DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_M / DIM_XA; m++)
                ra[n][m] = offs_dA[n * DIM_YA * LDA + m * DIM_XA];

        #pragma unroll
        for (n = 0; n < BLK_N / DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_K / DIM_XB; m++)
                rb[n][m] = offs_dB[n * DIM_YB * LDB + m * DIM_XB];

        // multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m * DIM_X + idx];
            
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n * DIM_Y + idy][k];

            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    rC[n][m] += rA[m] * rB[n];
                }
            }
        }
        __syncthreads();

        // store A regs->smem
        #pragma unroll
        for (n = 0; n < BLK_K / DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_M / DIM_XA; m++)
                sA[n * DIM_YA + idyA][m * DIM_XA + idxA] = ra[n][m];

        #pragma unroll
        for (n = 0; n < BLK_N / DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_K / DIM_XB; m++)
                sB[n * DIM_YB + idyB][m * DIM_XB + idxB] = rb[n][m];
        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of columns of op(A) and
    // rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.

    kk = K - kk;
    #pragma unroll
    for (k = 0; k < kk; k++)
    {
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m * DIM_X + idx];

        #pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n * DIM_Y + idy][k];
        
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++){
                rC[n][m] += rA[m] * rB[n];
            }
        }
    }

    #pragma unroll
    for (n = 0; n < THR_N; n++) {
        int coord_dCn = bly * BLK_N + n * DIM_Y + idy;
        #pragma unroll
        for (m = 0; m < THR_M; m++) {
            int coord_dCm = blx * BLK_M + m * DIM_X + idx;
            if (coord_dCm < M && coord_dCn < N) {
                C[coord_dCn * LDC + coord_dCm] = rC[n][m];
            }
        }
    }
}
