
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


def parks_mclellan_bp(n, freqs, amplitudes, weights, eps=0.01, nmax=4):
    breakpoint()
    band_idx = cupy.arange(freqs.shape[0])  # NOQA
    band_start = cupy.zeros(freqs.shape[0], dtype=cupy.bool_)

    block_sz = 128
    n_blocks = (band_start.shape[0] + block_sz - 1) // block_sz
    ker_band_len((n_blocks,), (block_sz,), (
        weights.shape[0], freqs, band_start))


def parks_mclellan(n, freqs, amplitudes, weights,
                   type='bandpass', eps=0.01, nmax=4):
    if type == 'bandpass':
        return parks_mclellan_bp(n, freqs, amplitudes, weights, eps, nmax)
