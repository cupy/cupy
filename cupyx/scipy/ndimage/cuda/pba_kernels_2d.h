// Euclidean Distance Transform
// 
// Kernels for the 2D version of the Parallel Banding Algorithm (PBA+). 
// 
// MIT license: see LICENSE in this folder
// Copyright: (c) 2019 School of Computing, National University of Singapore
//
// Modifications by Gregory Lee (2022) (NVIDIA)
// - add user-defined pixel_int2_t to enable
// - replace __mul24 operations with standard multiplication operator
// - Add variant kernels with support for non-isotropic pixel dimensions. These
//   kernels differ from the originals in that they also take sx and sy values
//   indicating the pixel size along the x and y axes. The kernels are identical
//   except that the `dominate` function is replaced by `dominate_sp` and the
//   physical spacings are used when computing distances.
//


// START OF DEFINITIONS OVERRIDDEN BY THE PYTHON SCRIPT

// The values included in this header file are those defined in the original
// PBA+ implementation

// However, the Python code generation can potentially generate a different
// ENCODE/DECODE that use 20 bits per coordinates instead of 10 bits per
// coordinate with ENCODED_INT_TYPE as `long long`.

#ifndef MARKER
#define MARKER -32768
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE 32
#endif

#ifndef pixel_int2_t
#define pixel_int2_t short2
#define make_pixel(x, y)  make_short2(x, y)
#endif

// END OF DEFINITIONS OVERRIDDEN BY THE PYTHON SCRIPT


#define TOID(x, y, size)  ((y) * (size) + (x))

#define LL long long
__device__ bool dominate(LL x1, LL y1, LL x2, LL y2, LL x3, LL y3, LL x0)
{
    LL k1 = y2 - y1, k2 = y3 - y2;
    return (k1 * (y1 + y2) + (x2 - x1) * ((x1 + x2) - (x0 << 1))) * k2 > \
            (k2 * (y2 + y3) + (x3 - x2) * ((x2 + x3) - (x0 << 1))) * k1;
}
#undef LL

// version of dominate, but with per-axis floating-point spacing
__device__ bool dominate_sp(int _x1, int _y1, int _x2, int _y2, int _x3, int _y3, int _x0, float sx, float sy)
{
    float x1 = static_cast<float>(_x1) * sx;
    float x2 = static_cast<float>(_x2) * sx;
    float x3 = static_cast<float>(_x3) * sx;
    float y1 = static_cast<float>(_y1) * sy;
    float y2 = static_cast<float>(_y2) * sy;
    float y3 = static_cast<float>(_y3) * sy;
    float x0_2 = static_cast<float>(_x0 << 1) * sx;
    float k1 = (y2 - y1);
    float k2 = (y3 - y2);
    return (k1 * (y1 + y2) + (x2 - x1) * ((x1 + x2) - x0_2)) * k2 > \
            (k2 * (y2 + y3) + (x3 - x2) * ((x2 + x3) - x0_2)) * k1;
}


extern "C"{

__global__ void kernelFloodDown(pixel_int2_t *input, pixel_int2_t *output, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * bandSize;
    int id = TOID(tx, ty, size);

    pixel_int2_t pixel1, pixel2;

    pixel1 = make_pixel(MARKER, MARKER);

    for (int i = 0; i < bandSize; i++, id += size) {
        pixel2 = input[id];

        if (pixel2.x != MARKER)
            pixel1 = pixel2;

        output[id] = pixel1;
    }
}

__global__ void kernelFloodUp(pixel_int2_t *input, pixel_int2_t *output, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = (blockIdx.y+1) * bandSize - 1;
    int id = TOID(tx, ty, size);

    pixel_int2_t pixel1, pixel2;
    int dist1, dist2;

    pixel1 = make_pixel(MARKER, MARKER);

    for (int i = 0; i < bandSize; i++, id -= size) {
        dist1 = abs(pixel1.y - ty + i);

        pixel2 = input[id];
        dist2 = abs(pixel2.y - ty + i);

        if (dist2 < dist1)
            pixel1 = pixel2;

        output[id] = pixel1;
    }
}

__global__ void kernelPropagateInterband(pixel_int2_t *input, pixel_int2_t *margin_out, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int inc = bandSize * size;
    int ny, nid, nDist;
    pixel_int2_t pixel;

    // Top row, look backward
    int ty = blockIdx.y * bandSize;
    int topId = TOID(tx, ty, size);
    int bottomId = TOID(tx, ty + bandSize - 1, size);
    int tid = blockIdx.y * size + tx;
    int bid = tid + (size * size / bandSize);

    pixel = input[topId];
    int myDist = abs(pixel.y - ty);
    margin_out[tid] = pixel;

    for (nid = bottomId - inc; nid >= 0; nid -= inc) {
        pixel = input[nid];

        if (pixel.x != MARKER) {
            nDist = abs(pixel.y - ty);

            if (nDist < myDist)
                margin_out[tid] = pixel;

            break;
        }
    }

    // Last row, look downward
    ty = ty + bandSize - 1;
    pixel = input[bottomId];
    myDist = abs(pixel.y - ty);
    margin_out[bid] = pixel;

    for (ny = ty + 1, nid = topId + inc; ny < size; ny += bandSize, nid += inc) {
        pixel = input[nid];

        if (pixel.x != MARKER) {
            nDist = abs(pixel.y - ty);

            if (nDist < myDist)
                margin_out[bid] = pixel;

            break;
        }
    }
}

__global__ void kernelUpdateVertical(pixel_int2_t *color, pixel_int2_t *margin, pixel_int2_t *output, int size, int bandSize)
{
    __shared__ pixel_int2_t block[BLOCKSIZE][BLOCKSIZE];

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * bandSize;

    pixel_int2_t top = margin[blockIdx.y * size + tx];
    pixel_int2_t bottom = margin[(blockIdx.y + size / bandSize) * size + tx];
    pixel_int2_t pixel;

    int dist, myDist;

    int id = TOID(tx, ty, size);

    int n_step = bandSize / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        int y_start = blockIdx.y * bandSize + step * blockDim.x;
        int y_end = y_start + blockDim.x;

        for (ty = y_start; ty < y_end; ++ty, id += size) {
            pixel = color[id];
            myDist = abs(pixel.y - ty);

            dist = abs(top.y - ty);
            if (dist < myDist) { myDist = dist; pixel = top; }

            dist = abs(bottom.y - ty);
            if (dist < myDist) pixel = bottom;

            // temporary result is stored in block
            block[threadIdx.x][ty - y_start] = make_pixel(pixel.y, pixel.x);
        }

        __syncthreads();

        // block is written to a transposed location in the output

        int tid = TOID(blockIdx.y * bandSize + step * blockDim.x + threadIdx.x, \
                        blockIdx.x * blockDim.x, size);

        for(int i = 0; i < blockDim.x; ++i, tid += size) {
            output[tid] = block[i][threadIdx.x];
        }

        __syncthreads();
    }
}

__global__ void kernelProximatePoints(pixel_int2_t *input, pixel_int2_t *stack, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * bandSize;
    int id = TOID(tx, ty, size);
    int lasty = -1;
    pixel_int2_t last1, last2, current;

    last1.y = -1; last2.y = -1;

    for (int i = 0; i < bandSize; i++, id += size) {
        current = input[id];

        if (current.x != MARKER) {
            while (last2.y >= 0) {
                if (!dominate(last1.x, last2.y, last2.x, \
                    lasty, current.x, current.y, tx))
                    break;

                lasty = last2.y; last2 = last1;

                if (last1.y >= 0)
                    last1 = stack[TOID(tx, last1.y, size)];
            }

            last1 = last2; last2 = make_pixel(current.x, lasty); lasty = current.y;

            stack[id] = last2;
        }
    }

    // Store the pointer to the tail at the last pixel of this band
    if (lasty != ty + bandSize - 1)
        stack[TOID(tx, ty + bandSize - 1, size)] = make_pixel(MARKER, lasty);
}


__global__ void kernelProximatePointsWithSpacing(pixel_int2_t *input, pixel_int2_t *stack, int size, int bandSize, double sx, double sy)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * bandSize;
    int id = TOID(tx, ty, size);
    int lasty = -1;
    pixel_int2_t last1, last2, current;

    last1.y = -1; last2.y = -1;

    for (int i = 0; i < bandSize; i++, id += size) {
        current = input[id];

        if (current.x != MARKER) {
            while (last2.y >= 0) {
                if (!dominate_sp(last1.x, last2.y, last2.x, \
                    lasty, current.x, current.y, tx, sx, sy))
                    break;

                lasty = last2.y; last2 = last1;

                if (last1.y >= 0)
                    last1 = stack[TOID(tx, last1.y, size)];
            }

            last1 = last2; last2 = make_pixel(current.x, lasty); lasty = current.y;

            stack[id] = last2;
        }
    }

    // Store the pointer to the tail at the last pixel of this band
    if (lasty != ty + bandSize - 1)
        stack[TOID(tx, ty + bandSize - 1, size)] = make_pixel(MARKER, lasty);
}

__global__ void kernelCreateForwardPointers(pixel_int2_t *input, pixel_int2_t *output, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = (blockIdx.y+1) * bandSize - 1;
    int id = TOID(tx, ty, size);
    int lasty = -1, nexty;
    pixel_int2_t current;

    // Get the tail pointer
    current = input[id];

    if (current.x == MARKER)
        nexty = current.y;
    else
        nexty = ty;

    for (int i = 0; i < bandSize; i++, id -= size)
        if (ty - i == nexty) {
            current = make_pixel(lasty, input[id].y);
            output[id] = current;

            lasty = nexty;
            nexty = current.y;
        }

    // Store the pointer to the head at the first pixel of this band
    if (lasty != ty - bandSize + 1)
        output[id + size] = make_pixel(lasty, MARKER);
}

__global__ void kernelMergeBands(pixel_int2_t *color, pixel_int2_t *link, pixel_int2_t *output, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int band1 = blockIdx.y * 2;
    int band2 = band1 + 1;
    int firsty, lasty;
    pixel_int2_t last1, last2, current;
    // last1 and last2: x component store the x coordinate of the site,
    // y component store the backward pointer
    // current: y component store the x coordinate of the site,
    // x component store the forward pointer

    // Get the two last items of the first list
    lasty = band2 * bandSize - 1;
    last2 = make_pixel(color[TOID(tx, lasty, size)].x,
        link[TOID(tx, lasty, size)].y);

    if (last2.x == MARKER) {
        lasty = last2.y;

        if (lasty >= 0)
            last2 = make_pixel(color[TOID(tx, lasty, size)].x,
            link[TOID(tx, lasty, size)].y);
        else
            last2 = make_pixel(MARKER, MARKER);
    }

    if (last2.y >= 0) {
        // Second item at the top of the stack
        last1 = make_pixel(color[TOID(tx, last2.y, size)].x,
            link[TOID(tx, last2.y, size)].y);
    }

    // Get the first item of the second band
    firsty = band2 * bandSize;
    current = make_pixel(link[TOID(tx, firsty, size)].x,
        color[TOID(tx, firsty, size)].x);

    if (current.y == MARKER) {
        firsty = current.x;

        if (firsty >= 0)
            current = make_pixel(link[TOID(tx, firsty, size)].x,
            color[TOID(tx, firsty, size)].x);
        else
            current = make_pixel(MARKER, MARKER);
    }

    // Count the number of item in the second band that survive so far.
    // Once it reaches 2, we can stop.
    int top = 0;

    while (top < 2 && current.y >= 0) {
        // While there's still something on the left
        while (last2.y >= 0) {

            if (!dominate(last1.x, last2.y, last2.x, \
                lasty, current.y, firsty, tx))
                break;

            lasty = last2.y; last2 = last1;
            top--;

            if (last1.y >= 0)
                last1 = make_pixel(color[TOID(tx, last1.y, size)].x,
                link[TOID(tx, last1.y, size)].y);
        }

        // Update the current pointer
        output[TOID(tx, firsty, size)] = make_pixel(current.x, lasty);

        if (lasty >= 0)
            output[TOID(tx, lasty, size)] = make_pixel(firsty, last2.y);

        last1 = last2; last2 = make_pixel(current.y, lasty); lasty = firsty;
        firsty = current.x;

        top = max(1, top + 1);

        // Advance the current pointer to the next one
        if (firsty >= 0)
            current = make_pixel(link[TOID(tx, firsty, size)].x,
            color[TOID(tx, firsty, size)].x);
        else
            current = make_pixel(MARKER, MARKER);
    }

    // Update the head and tail pointer.
    firsty = band1 * bandSize;
    lasty = band2 * bandSize;
    current = link[TOID(tx, firsty, size)];

    if (current.y == MARKER && current.x < 0) { // No head?
        last1 = link[TOID(tx, lasty, size)];

        if (last1.y == MARKER)
            current.x = last1.x;
        else
            current.x = lasty;

        output[TOID(tx, firsty, size)] = current;
    }

    firsty = band1 * bandSize + bandSize - 1;
    lasty = band2 * bandSize + bandSize - 1;
    current = link[TOID(tx, lasty, size)];

    if (current.x == MARKER && current.y < 0) { // No tail?
        last1 = link[TOID(tx, firsty, size)];

        if (last1.x == MARKER)
            current.y = last1.y;
        else
            current.y = firsty;

        output[TOID(tx, lasty, size)] = current;
    }
}


__global__ void kernelMergeBandsWithSpacing(pixel_int2_t *color, pixel_int2_t *link, pixel_int2_t *output, int size, int bandSize, double sx, double sy)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int band1 = blockIdx.y * 2;
    int band2 = band1 + 1;
    int firsty, lasty;
    pixel_int2_t last1, last2, current;
    // last1 and last2: x component store the x coordinate of the site,
    // y component store the backward pointer
    // current: y component store the x coordinate of the site,
    // x component store the forward pointer

    // Get the two last items of the first list
    lasty = band2 * bandSize - 1;
    last2 = make_pixel(color[TOID(tx, lasty, size)].x,
        link[TOID(tx, lasty, size)].y);

    if (last2.x == MARKER) {
        lasty = last2.y;

        if (lasty >= 0)
            last2 = make_pixel(color[TOID(tx, lasty, size)].x,
            link[TOID(tx, lasty, size)].y);
        else
            last2 = make_pixel(MARKER, MARKER);
    }

    if (last2.y >= 0) {
        // Second item at the top of the stack
        last1 = make_pixel(color[TOID(tx, last2.y, size)].x,
            link[TOID(tx, last2.y, size)].y);
    }

    // Get the first item of the second band
    firsty = band2 * bandSize;
    current = make_pixel(link[TOID(tx, firsty, size)].x,
        color[TOID(tx, firsty, size)].x);

    if (current.y == MARKER) {
        firsty = current.x;

        if (firsty >= 0)
            current = make_pixel(link[TOID(tx, firsty, size)].x,
            color[TOID(tx, firsty, size)].x);
        else
            current = make_pixel(MARKER, MARKER);
    }

    // Count the number of item in the second band that survive so far.
    // Once it reaches 2, we can stop.
    int top = 0;

    while (top < 2 && current.y >= 0) {
        // While there's still something on the left
        while (last2.y >= 0) {

            if (!dominate_sp(last1.x, last2.y, last2.x, \
                lasty, current.y, firsty, tx, sx, sy))
                break;

            lasty = last2.y; last2 = last1;
            top--;

            if (last1.y >= 0)
                last1 = make_pixel(color[TOID(tx, last1.y, size)].x,
                link[TOID(tx, last1.y, size)].y);
        }

        // Update the current pointer
        output[TOID(tx, firsty, size)] = make_pixel(current.x, lasty);

        if (lasty >= 0)
            output[TOID(tx, lasty, size)] = make_pixel(firsty, last2.y);

        last1 = last2; last2 = make_pixel(current.y, lasty); lasty = firsty;
        firsty = current.x;

        top = max(1, top + 1);

        // Advance the current pointer to the next one
        if (firsty >= 0)
            current = make_pixel(link[TOID(tx, firsty, size)].x,
            color[TOID(tx, firsty, size)].x);
        else
            current = make_pixel(MARKER, MARKER);
    }

    // Update the head and tail pointer.
    firsty = band1 * bandSize;
    lasty = band2 * bandSize;
    current = link[TOID(tx, firsty, size)];

    if (current.y == MARKER && current.x < 0) { // No head?
        last1 = link[TOID(tx, lasty, size)];

        if (last1.y == MARKER)
            current.x = last1.x;
        else
            current.x = lasty;

        output[TOID(tx, firsty, size)] = current;
    }

    firsty = band1 * bandSize + bandSize - 1;
    lasty = band2 * bandSize + bandSize - 1;
    current = link[TOID(tx, lasty, size)];

    if (current.x == MARKER && current.y < 0) { // No tail?
        last1 = link[TOID(tx, firsty, size)];

        if (last1.x == MARKER)
            current.y = last1.y;
        else
            current.y = firsty;

        output[TOID(tx, lasty, size)] = current;
    }
}

__global__ void kernelDoubleToSingleList(pixel_int2_t *color, pixel_int2_t *link, pixel_int2_t *output, int size)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y;
    int id = TOID(tx, ty, size);

    output[id] = make_pixel(color[id].x, link[id].y);
}

__global__ void kernelColor(pixel_int2_t *input, pixel_int2_t *output, int size)
{
    __shared__ pixel_int2_t block[BLOCKSIZE][BLOCKSIZE];

    int col = threadIdx.x;
    int tid = threadIdx.y;
    int tx = blockIdx.x * blockDim.x + col;
    int dx, dy, lasty;
    unsigned int best, dist;
    pixel_int2_t last1, last2;

    lasty = size - 1;

    last2 = input[TOID(tx, lasty, size)];

    if (last2.x == MARKER) {
        lasty = max(last2.y, 0);
        last2 = input[TOID(tx, lasty, size)];
    }

    if (last2.y >= 0)
        last1 = input[TOID(tx, last2.y, size)];

    int y_start, y_end, n_step = size / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        y_start = size - step * blockDim.x - 1;
        y_end = size - (step + 1) * blockDim.x;

        for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
            dx = last2.x - tx; dy = lasty - ty;
            best = dist = dx * dx + dy * dy;

            while (last2.y >= 0) {
                dx = last1.x - tx; dy = last2.y - ty;
                dist = dx * dx + dy * dy;

                if (dist > best)
                    break;

                best = dist; lasty = last2.y; last2 = last1;

                if (last2.y >= 0)
                    last1 = input[TOID(tx, last2.y, size)];
            }

            block[threadIdx.x][ty - y_end] = make_pixel(lasty, last2.x);
        }

        __syncthreads();

        // note: transposes back to original shape here
        if(!threadIdx.y) {
            int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x, size);
            for(int i = 0; i < blockDim.x; ++i, id+=size) {
                output[id] = block[i][threadIdx.x];
            }
        }

        __syncthreads();
    }
}


__global__ void kernelColorWithSpacing(pixel_int2_t *input, pixel_int2_t *output, int size, double sx, double sy)
{
    __shared__ pixel_int2_t block[BLOCKSIZE][BLOCKSIZE];

    int col = threadIdx.x;
    int tid = threadIdx.y;
    int tx = blockIdx.x * blockDim.x + col;
    int lasty;
    double dx, dy, best, dist;
    pixel_int2_t last1, last2;

    lasty = size - 1;

    last2 = input[TOID(tx, lasty, size)];

    if (last2.x == MARKER) {
        lasty = max(last2.y, 0);
        last2 = input[TOID(tx, lasty, size)];
    }

    if (last2.y >= 0)
        last1 = input[TOID(tx, last2.y, size)];

    int y_start, y_end, n_step = size / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        y_start = size - step * blockDim.x - 1;
        y_end = size - (step + 1) * blockDim.x;

        for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
            dx = static_cast<double>(last2.x - tx) * sx;
            dy = static_cast<double>(lasty - ty) * sy;
            best = dist = dx * dx + dy * dy;

            while (last2.y >= 0) {
                dx = static_cast<double>(last1.x - tx) * sx;
                dy = static_cast<double>(last2.y - ty) * sy;
                dist = dx * dx + dy * dy;

                if (dist > best)
                    break;

                best = dist; lasty = last2.y; last2 = last1;

                if (last2.y >= 0)
                    last1 = input[TOID(tx, last2.y, size)];
            }

            block[threadIdx.x][ty - y_end] = make_pixel(lasty, last2.x);
        }

        __syncthreads();

        // note: transposes back to original shape here
        if(!threadIdx.y) {
            int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x, size);
            for(int i = 0; i < blockDim.x; ++i, id+=size) {
                output[id] = block[i][threadIdx.x];
            }
        }

        __syncthreads();
    }
}
} // extern C
