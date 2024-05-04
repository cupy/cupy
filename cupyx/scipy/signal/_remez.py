
import cupy  # NOQA


ker_band_len = cupy.RawKernel(r'''
extern "C" __global__ void ker_band_len(
    int n_weights, const double* freqs, bool* band_start
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= n_weights) {
        return;
    }

    if(idx == 0) {
        band_start[idx] = true;
    }

    if(idx < n_weights - 1) {
        band_start[idx + 1] = freqs[2 * idx + 1] != freqs[2 * idx + 2];
    }
}
''', 'ker_band_len')


ker_t12_band_init = cupy.RawKernel(r'''
#define CUDART_PI  3.1415926535897931e+0

__device__ int find_band(
        const int item, const int n_bands, const long long* band_map) {
    int left = 0;
    int right = n_bands;

    while(left < right) {
        int mid = (left + right) / 2;
        if(band_map[mid] < item) {
            left = mid + 1;
        } else if (band_map[mid] > item) {
            right = mid;
        } else {
            left = mid + 1;
            break;
        }
    }

    return left;
}


extern "C" __global__ void ker_t12_band_init(
    int n_band_items, int n_bands, bool t2, const long long* band_map,
    const long long* band_start, const double* freqs,
    const long long* fbands_start, double* fbands, long long* freq_bands
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= n_band_items) {
        return;
    }

    int band = find_band(idx, n_bands, band_map);

    if(idx == band_start[band]) {
        fbands[idx + band] = CUDART_PI * freqs[2 * band];
        freq_bands[idx + band] = band;
    }

    if(t2 && freqs[2 * idx + 1] == 1.0) {
        if(freqs[2 * idx] < 0.9999) {
            fbands[idx + band + 1] = CUDART_PI * 0.9999;
        } else {
            fbands[idx + band + 1] = CUDART_PI * (freqs[2 * idx] + 1) / 2;
        }
    } else {
        fbands[idx + band + 1] = CUDART_PI * freqs[2 * idx + 1];
    }

    freq_bands[idx + band + 1] = band;

}
''', 'ker_t12_band_init')


ker_bandconv = cupy.RawKernel(r'''
extern "C" __global__ void bandconv(
    int n, bool inverse, const long long* freq_bands,
    const long long* band_bounds, const double* in,
    double* out, long long* mapping, bool* out_space
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= n) {
        return;
    }

    int inv_pos = n - 1 - idx;
    int band = freq_bands[inv_pos];
    int band_bound = band_bounds[band];
    int act_pos = band_bound - 1 - inv_pos;
    int band_start = band == 0 ? 0 : band_bounds[band - 1];

    if(inverse) {
        out[idx] = acos(in[act_pos + band_start]);
    } else {
        out[idx] = cos(in[act_pos + band_start]);
    }
    mapping[idx] = act_pos + band_start;
    out_space[idx] = !out_space[idx];
}
''', 'bandconv')


ker_sort_bands = cupy.RawKernel(r'''
extern "C" __global__ void sort_bands(
    const long long* band_length, const long long* band_bounds, double* cbands
) {
    int nband = blockIdx.x;
    int pos = threadIdx.x;

    const long long length = band_length[nband];
    const long long start = band_bounds[nband] - length;

    if(pos >= length) {
        return;
    }

    double* band = cbands + start;
    long long l = length % 2 == 0 ? length / 2 : (length / 2) + 1;

    for(long long i = 0; i < l; i++) {
        if(!(pos & 1) && pos < (length - 1)) {
            if(band[pos] > band[pos + 1]) {
                double temp = band[pos];
                band[pos] = band[pos + 1];
                band[pos + 1] = temp;
            }
        }

        __syncthreads();

        if((pos & 1) && pos < (length - 1)) {
            if(band[pos] > band[pos + 1]) {
                double temp = band[pos];
                band[pos] = band[pos + 1];
                band[pos + 1] = temp;
            }
        }

        __syncthreads();
    }
}
''', 'sort_bands')


ker_bandwidths = cupy.RawKernel(r'''
extern "C" __global__ void ker_bandwidths(
    int n, const double* freqs, const long long* band_length,
    const long long* band_end, double* bandwidths, long long* xs
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= n) {
        return;
    }

    const long long length = band_length[idx];
    const long long end = band_end[idx] - 1;
    const long long start = end - length;

    bandwidths[idx] = freqs[end] - freqs[start];
    xs[idx] = 1;
}
''', 'ker_bandwidths')


def parks_mclellan_bp(n, freqs, amplitudes, weights, eps=0.01, nmax=4):
    band_idx = cupy.arange(weights.shape[0])
    band_mask = cupy.zeros(weights.shape[0], dtype=cupy.bool_)

    block_sz = 128
    n_blocks = (band_mask.shape[0] + block_sz - 1) // block_sz
    ker_band_len((n_blocks,), (block_sz,), (
        weights.shape[0], freqs, band_mask))

    nbands = band_mask.sum()

    band_start = band_idx[band_mask]
    band_length = band_start.copy()

    band_length[:-1] = band_start[1:] - band_length[:-1]
    band_length[-1] = weights.shape[0] - band_length[-1]
    band_map = cupy.cumsum(band_length)
    fbands_displ = cupy.cumsum(band_length + 1)

    nall_bands = band_length.sum().item()

    fbands = cupy.empty(nall_bands + band_start.shape[0], dtype=cupy.float64)
    freq_bands = cupy.empty_like(fbands, dtype=cupy.int64)
    breakpoint()

    is_type_2 = n % 2 != 0
    n_blocks = (nall_bands + block_sz - 1) // block_sz
    ker_t12_band_init((n_blocks,), (block_sz,), (
        nall_bands, nbands.item(), is_type_2, band_map, band_start,
        freqs, fbands_displ, fbands, freq_bands))

    fspace = cupy.zeros_like(fbands, dtype=cupy.bool_)
    cmapping = cupy.empty_like(fbands, dtype=cupy.int64)
    cbands = cupy.empty_like(fbands)
    cspace = cupy.zeros_like(fspace)

    n_blocks = (fbands.shape[0] + block_sz - 1) // block_sz
    ker_bandconv((n_blocks,), (block_sz,), (
        fbands.shape[0], False, freq_bands, fbands_displ, fbands,
        cbands, cmapping, cspace))

    cbands_length = band_length[::-1].copy() + 1
    cbands_displ = cupy.cumsum(cbands_length)

    ker_sort_bands((cbands_length.shape[0],), (block_sz,), (
        cbands_length, cbands_displ, cbands))

    bandwidths = cupy.empty(band_length.shape[0], dtype=cupy.float64)
    xs = cupy.empty_like(bandwidths, dtype=cupy.int64)

    n_blocks = (band_length.shape[0] + block_sz - 1) // block_sz
    ker_bandwidths((n_blocks,), (block_sz,), (
        band_length.shape[0], fbands, band_length, fbands_displ,
        bandwidths, xs))

    omega = cupy.empty(n // 2 + 2, dtype=cupy.float64)
    non_point_bands = cupy.where(bandwidths > 0.0)[0]
    avg_dist = (  # NOQA
        bandwidths[non_point_bands].sum() / (
            omega.shape[0] - bandwidths.shape[0]))  # NOQA


def parks_mclellan(n, freqs, amplitudes, weights,
                   type='bandpass', eps=0.01, nmax=4):
    if type == 'bandpass':
        return parks_mclellan_bp(n, freqs, amplitudes, weights, eps, nmax)
