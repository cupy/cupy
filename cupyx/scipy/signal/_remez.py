
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


ker_t2_band_init = cupy.RawKernel(r'''
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


extern "C" __global__ void ker_t2_band_init(
    int n_band_items, int n_bands, const long long* band_map,
    const double* freqs, const long long* fbands_start, double* fbands
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= n_band_items) {
        return;
    }

    int band = find_band(idx, n_bands, band_map);
    idx += band;

    if(idx == band_map[band]) {
        fbands[idx] = CUDART_PI * freqs[2 * band];
    }

    if(freqs[2 * idx + 1] == 1.0) {
        if(freqs[2 * idx] < 0.9999) {
            fbands[idx + 1] = CUDART_PI * 0.9999;
        } else {
            fbands[idx + 1] = CUDART_PI * (freqs[2 * idx] + 1) / 2;
        }
    } else {
        fbands[idx + 1] = CUDART_PI * freqs[2 * idx + 1];
    }

}
''', 'ker_t2_band_init')


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
    breakpoint()
    if n % 2 == 0:
        # Type I filter
        pass
    else:
        # Type II filter
        n_blocks = (nall_bands + block_sz - 1) // block_sz
        ker_t2_band_init((n_blocks,), (block_sz,), (
            nall_bands, nbands.item(), band_map, freqs, fbands_displ, fbands))


def parks_mclellan(n, freqs, amplitudes, weights,
                   type='bandpass', eps=0.01, nmax=4):
    if type == 'bandpass':
        return parks_mclellan_bp(n, freqs, amplitudes, weights, eps, nmax)
