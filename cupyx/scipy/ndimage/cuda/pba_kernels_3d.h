// Euclidean Distance Transform
// 
// Kernels for the 3D version of the Parallel Banding Algorithm (PBA+). 
// 
// MIT license: see LICENSE in this folder
// Copyright: (c) 2019 School of Computing, National University of Singapore
//
// Modifications by Gregory Lee (2022) (NVIDIA)
// - allow user-defined ENCODED_INT_TYPE, ENCODE, DECODE
// - Add variant kernels with support for non-isotropic pixel dimensions
//   These kernels differ from the originals in that they also take sx, sy and
//   sz values indicating the pixel size along the x, y and z axes. The kernels
//   are identical except that the `dominate` function is replaced by
//   `dominate_sp` and the physical spacings are used when computing distances.


// START OF DEFINITIONS OVERRIDDEN BY THE PYTHON SCRIPT

// The values included in this header file are those defined in the original
// PBA+ implementation

// However, the Python code generation can potentially generate a different
// ENCODE/DECODE that use 20 bits per coordinates instead of 10 bits per
// coordinate with ENCODED_INT_TYPE as `long long`.


#ifndef MARKER
#define MARKER     -2147483648
#endif  // MARKER

#ifndef MAX_INT
#define MAX_INT    2147483647
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE  32
#endif

#ifndef ENCODE

// Sites     : ENCODE(x, y, z, 0, 0)
// Not sites : ENCODE(0, 0, 0, 1, 0) or MARKER
#define ENCODED_INT_TYPE int
#define ZERO 0
#define ONE 1
#define ENCODE(x, y, z, a, b)  (((x) << 20) | ((y) << 10) | (z) | ((a) << 31) | ((b) << 30))
#define DECODE(value, x, y, z) \
    x = ((value) >> 20) & 0x3ff; \
    y = ((value) >> 10) & 0x3ff; \
    z = (value) & 0x3ff

#define NOTSITE(value)  (((value) >> 31) & 1)
#define HASNEXT(value)  (((value) >> 30) & 1)

#define GET_X(value)    (((value) >> 20) & 0x3ff)
#define GET_Y(value)    (((value) >> 10) & 0x3ff)
#define GET_Z(value)    ((NOTSITE((value))) ? MAX_INT : ((value) & 0x3ff))

#endif // ENCODE

// END OF DEFINITIONS DEFINED IN THE PYTHON SCRIPT


#define LL long long
__device__ bool dominate(LL x_1, LL y_1, LL z_1, LL x_2, LL y_2, LL z_2, LL x_3, LL y_3, LL z_3, LL x_0, LL z_0)
{
    LL k_1 = y_2 - y_1, k_2 = y_3 - y_2;

    return (((y_1 + y_2) * k_1 + ((x_2 - x_1) * (x_1 + x_2 - (x_0 << 1)) + (z_2 - z_1) * (z_1 + z_2 - (z_0 << 1)))) * k_2 > \
            ((y_2 + y_3) * k_2 + ((x_3 - x_2) * (x_2 + x_3 - (x_0 << 1)) + (z_3 - z_2) * (z_2 + z_3 - (z_0 << 1)))) * k_1);
}
#undef LL

__device__ bool dominate_sp(int _x_1, int _y_1, int _z_1, int _x_2, int _y_2, int _z_2, int _x_3, int _y_3, int _z_3, int _x_0, int _z_0, float sx, float sy, float sz)
{
    float x_1 = static_cast<float>(_x_1) * sx;
    float y_1 = static_cast<float>(_y_1) * sy;
    float z_1 = static_cast<float>(_z_1) * sz;
    float x_2 = static_cast<float>(_x_2) * sx;
    float y_2 = static_cast<float>(_y_2) * sy;
    float z_2 = static_cast<float>(_z_2) * sz;
    float x_3 = static_cast<float>(_x_3) * sx;
    float y_3 = static_cast<float>(_y_3) * sy;
    float z_3 = static_cast<float>(_z_3) * sz;
    float x_0_2 = static_cast<float>(_x_0 << 1) * sx;
    float z_0_2 = static_cast<float>(_z_0 << 1) * sz;
    float k_1 = y_2 - y_1;
    float k_2 = y_3 - y_2;

    return (((y_1 + y_2) * k_1 + ((x_2 - x_1) * (x_1 + x_2 - (x_0_2)) + (z_2 - z_1) * (z_1 + z_2 - (z_0_2)))) * k_2 > \
            ((y_2 + y_3) * k_2 + ((x_3 - x_2) * (x_2 + x_3 - (x_0_2)) + (z_3 - z_2) * (z_2 + z_3 - (z_0_2)))) * k_1);
}

#define TOID(x, y, z, size)    ((((z) * (size)) + (y)) * (size) + (x))


extern "C"{

__global__ void kernelFloodZ(ENCODED_INT_TYPE *input, ENCODED_INT_TYPE *output, int size)
{

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = 0;

    int plane = size * size;
    int id = TOID(tx, ty, tz, size);
    ENCODED_INT_TYPE pixel1, pixel2;

    pixel1 = ENCODE(ZERO,ZERO,ZERO,ONE,ZERO);

    // Sweep down
    for (int i = 0; i < size; i++, id += plane) {
        pixel2 = input[id];

        if (!NOTSITE(pixel2))
            pixel1 = pixel2;

        output[id] = pixel1;
    }

    ENCODED_INT_TYPE dist1, dist2, nz;

    id -= plane + plane;

    // Sweep up
    for (int i = size - 2; i >= 0; i--, id -= plane) {
        nz = GET_Z(pixel1);
        dist1 = abs(nz - (tz + i));

        pixel2 = output[id];
        nz = GET_Z(pixel2);
        dist2 = abs(nz - (tz + i));

        if (dist2 < dist1)
            pixel1 = pixel2;

        output[id] = pixel1;
    }
}


__global__ void kernelMaurerAxis(ENCODED_INT_TYPE *input, ENCODED_INT_TYPE *stack, int size)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int tz = blockIdx.y * blockDim.y + threadIdx.y;
    int ty = 0;

    int id = TOID(tx, ty, tz, size);

    ENCODED_INT_TYPE lasty = 0;
    ENCODED_INT_TYPE x1, y1, z1, x2, y2, z2, nx, ny, nz;
    ENCODED_INT_TYPE p = ENCODE(ZERO,ZERO,ZERO,ONE,ZERO), s1 = ENCODE(ZERO,ZERO,ZERO,ONE,ZERO), s2 = ENCODE(ZERO,ZERO,ZERO,ONE,ZERO);
    ENCODED_INT_TYPE flag = 0;

    for (ty = 0; ty < size; ++ty, id += size) {
        p = input[id];

        if (!NOTSITE(p)) {

            while (HASNEXT(s2)) {
                DECODE(s1, x1, y1, z1);
                DECODE(s2, x2, y2, z2);
                DECODE(p, nx, ny, nz);

                if (!dominate(x1, y2, z1, x2, lasty, z2, nx, ty, nz, tx, tz))
                    break;

                lasty = y2; s2 = s1; y2 = y1;

                if (HASNEXT(s2))
                    s1 = stack[TOID(tx, y2, tz, size)];
            }

            DECODE(p, nx, ny, nz);
            s1 = s2;
            s2 = ENCODE(nx, lasty, nz, ZERO, flag);
            y2 = lasty;
            lasty = ty;

            stack[id] = s2;

            flag = ONE;
        }
    }

    if (NOTSITE(p))
        stack[TOID(tx, ty - 1, tz, size)] = ENCODE(ZERO, lasty, ZERO, ONE, flag);
}

__global__ void kernelMaurerAxisWithSpacing(ENCODED_INT_TYPE *input, ENCODED_INT_TYPE *stack, int size, double sx, double sy, double sz)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int tz = blockIdx.y * blockDim.y + threadIdx.y;
    int ty = 0;

    int id = TOID(tx, ty, tz, size);

    ENCODED_INT_TYPE lasty = 0;
    ENCODED_INT_TYPE x1, y1, z1, x2, y2, z2, nx, ny, nz;
    ENCODED_INT_TYPE p = ENCODE(ZERO,ZERO,ZERO,ONE,ZERO), s1 = ENCODE(ZERO,ZERO,ZERO,ONE,ZERO), s2 = ENCODE(ZERO,ZERO,ZERO,ONE,ZERO);
    ENCODED_INT_TYPE flag = 0;

    for (ty = 0; ty < size; ++ty, id += size) {
        p = input[id];

        if (!NOTSITE(p)) {

            while (HASNEXT(s2)) {
                DECODE(s1, x1, y1, z1);
                DECODE(s2, x2, y2, z2);
                DECODE(p, nx, ny, nz);

                if (!dominate_sp(x1, y2, z1, x2, lasty, z2, nx, ty, nz, tx, tz, sx, sy, sz))
                    break;

                lasty = y2; s2 = s1; y2 = y1;

                if (HASNEXT(s2))
                    s1 = stack[TOID(tx, y2, tz, size)];
            }

            DECODE(p, nx, ny, nz);
            s1 = s2;
            s2 = ENCODE(nx, lasty, nz, ZERO, flag);
            y2 = lasty;
            lasty = ty;

            stack[id] = s2;

            flag = ONE;
        }
    }

    if (NOTSITE(p))
        stack[TOID(tx, ty - 1, tz, size)] = ENCODE(ZERO, lasty, ZERO, ONE, flag);
}

__global__ void kernelColorAxis(ENCODED_INT_TYPE *input, ENCODED_INT_TYPE *output, int size)
{
    __shared__ ENCODED_INT_TYPE block[BLOCKSIZE][BLOCKSIZE];

    int col = threadIdx.x;
    int tid = threadIdx.y;
    int tx = blockIdx.x * blockDim.x + col;
    int tz = blockIdx.y;

    ENCODED_INT_TYPE x1, y1, z1, x2, y2, z2;
    ENCODED_INT_TYPE last1 = ENCODE(ZERO,ZERO,ZERO,ONE,ZERO), last2 = ENCODE(ZERO,ZERO,ZERO,ONE,ZERO), lasty;
    long long dx, dy, dz, best, dist;

    lasty = size - 1;

    last2 = input[TOID(tx, lasty, tz, size)];
    DECODE(last2, x2, y2, z2);

    if (NOTSITE(last2)) {
        lasty = y2;
        if(HASNEXT(last2)) {
            last2 = input[TOID(tx, lasty, tz, size)];
            DECODE(last2, x2, y2, z2);
        }
    }

    if (HASNEXT(last2)) {
        last1 = input[TOID(tx, y2, tz, size)];
        DECODE(last1, x1, y1, z1);
    }

    int y_start, y_end, n_step = size / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        y_start = size - step * blockDim.x - 1;
        y_end = size - (step + 1) * blockDim.x;

        for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
            dx = x2 - tx; dy = lasty - ty; dz = z2 - tz;
            best = dx * dx + dy * dy + dz * dz;

            while (HASNEXT(last2)) {
                dx = x1 - tx; dy = y2 - ty; dz = z1 - tz;
                dist = dx * dx + dy * dy + dz * dz;

                if(dist > best) break;

                best = dist; lasty = y2; last2 = last1;
                DECODE(last2, x2, y2, z2);

                if (HASNEXT(last2)) {
                    last1 = input[TOID(tx, y2, tz, size)];
                    DECODE(last1, x1, y1, z1);
                }
            }

            block[threadIdx.x][ty - y_end] = ENCODE(lasty, x2, z2, NOTSITE(last2), ZERO);
        }

        __syncthreads();

        if(!threadIdx.y) {
            int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x, tz, size);
            for(int i = 0; i < blockDim.x; i++, id+=size) {
                output[id] = block[i][threadIdx.x];
            }
        }

        __syncthreads();
    }
}


__global__ void kernelColorAxisWithSpacing(ENCODED_INT_TYPE *input, ENCODED_INT_TYPE *output, int size, double sx, double sy, double sz)
{
    __shared__ ENCODED_INT_TYPE block[BLOCKSIZE][BLOCKSIZE];

    int col = threadIdx.x;
    int tid = threadIdx.y;
    int tx = blockIdx.x * blockDim.x + col;
    int tz = blockIdx.y;

    ENCODED_INT_TYPE x1, y1, z1, x2, y2, z2;
    ENCODED_INT_TYPE last1 = ENCODE(ZERO,ZERO,ZERO,ONE,ZERO), last2 = ENCODE(ZERO,ZERO,ZERO,ONE,ZERO), lasty;
    double dx, dy, dz, best, dist;

    lasty = size - 1;

    last2 = input[TOID(tx, lasty, tz, size)];
    DECODE(last2, x2, y2, z2);

    if (NOTSITE(last2)) {
        lasty = y2;
        if(HASNEXT(last2)) {
            last2 = input[TOID(tx, lasty, tz, size)];
            DECODE(last2, x2, y2, z2);
        }
    }

    if (HASNEXT(last2)) {
        last1 = input[TOID(tx, y2, tz, size)];
        DECODE(last1, x1, y1, z1);
    }

    int y_start, y_end, n_step = size / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        y_start = size - step * blockDim.x - 1;
        y_end = size - (step + 1) * blockDim.x;

        for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
            dx = static_cast<double>(x2 - tx) * sx;
            dy = static_cast<double>(lasty - ty) * sy;
            dz = static_cast<double>(z2 - tz) * sz;
            best = dx * dx + dy * dy + dz * dz;

            while (HASNEXT(last2)) {
                dx = static_cast<double>(x1 - tx) * sx;
                dy = static_cast<double>(y2 - ty) * sy;
                dz = static_cast<double>(z1 - tz) * sz;
                dist = dx * dx + dy * dy + dz * dz;

                if(dist > best) break;

                best = dist; lasty = y2; last2 = last1;
                DECODE(last2, x2, y2, z2);

                if (HASNEXT(last2)) {
                    last1 = input[TOID(tx, y2, tz, size)];
                    DECODE(last1, x1, y1, z1);
                }
            }

            block[threadIdx.x][ty - y_end] = ENCODE(lasty, x2, z2, NOTSITE(last2), ZERO);
        }

        __syncthreads();

        if(!threadIdx.y) {
            int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x, tz, size);
            for(int i = 0; i < blockDim.x; i++, id+=size) {
                output[id] = block[i][threadIdx.x];
            }
        }

        __syncthreads();
    }
}


} // extern C
