
import cupy
from cupy_backends.cuda.api import runtime

_mls_taps = {2: [1], 3: [2], 4: [3], 5: [3], 6: [5], 7: [6], 8: [7, 6, 1],
             9: [5], 10: [7], 11: [9], 12: [11, 10, 4], 13: [12, 11, 8],
             14: [13, 12, 2], 15: [14], 16: [15, 13, 4], 17: [14],
             18: [11], 19: [18, 17, 14], 20: [17], 21: [19], 22: [21],
             23: [18], 24: [23, 22, 17], 25: [22], 26: [25, 24, 20],
             27: [26, 25, 22], 28: [25], 29: [27], 30: [29, 28, 7],
             31: [28], 32: [31, 30, 10]}

if runtime.is_hip:
    MAX_LEN_SEQ_BASE = r"""
    #include <hip/hip_runtime.h>
"""
else:
    MAX_LEN_SEQ_BASE = r"""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
"""

MAX_LEN_SEQ_KERNEL = MAX_LEN_SEQ_BASE + r"""
#include <cupy/math_constants.h>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

extern "C" __global__ void max_len_seq(
        long long length, long long n_taps, int n_state, long long* taps,
        signed char* state, signed char* seq) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for(long long i = 0; i < length; i++) {
        signed char next_state = state[(idx + 1) % n_state];
        if(idx == n_state - 1) {
            seq[i] = state[0];
            for(int n_tap = 0; n_tap < n_taps; n_tap++) {
                long long tap = taps[n_tap];
                next_state ^= state[tap];
            }
        }
        state[idx] = next_state;
    }
}
"""

_max_len_seq = cupy.RawKernel(MAX_LEN_SEQ_KERNEL, 'max_len_seq')


def max_len_seq(nbits, state=None, length=None, taps=None):
    """
    Maximum length sequence (MLS) generator.

    Parameters
    ----------
    nbits : int
        Number of bits to use. Length of the resulting sequence will
        be ``(2**nbits) - 1``. Note that generating long sequences
        (e.g., greater than ``nbits == 16``) can take a long time.
    state : array_like, optional
        If array, must be of length ``nbits``, and will be cast to binary
        (bool) representation. If None, a seed of ones will be used,
        producing a repeatable representation. If ``state`` is all
        zeros, an error is raised as this is invalid. Default: None.
    length : int, optional
        Number of samples to compute. If None, the entire length
        ``(2**nbits) - 1`` is computed.
    taps : array_like, optional
        Polynomial taps to use (e.g., ``[7, 6, 1]`` for an 8-bit sequence).
        If None, taps will be automatically selected (for up to
        ``nbits == 32``).

    Returns
    -------
    seq : array
        Resulting MLS sequence of 0's and 1's.
    state : array
        The final state of the shift register.

    Notes
    -----
    The algorithm for MLS generation is generically described in:

        https://en.wikipedia.org/wiki/Maximum_length_sequence

    The default values for taps are specifically taken from the first
    option listed for each value of ``nbits`` in:

        https://web.archive.org/web/20181001062252/http://www.newwaveinstruments.com/resources/articles/m_sequence_linear_feedback_shift_register_lfsr.htm

    """  # NOQA
    if taps is None:
        if nbits not in _mls_taps:
            known_taps = cupy.array(list(_mls_taps.keys()))
            raise ValueError('nbits must be between %s and %s if taps is None'
                             % (known_taps.min(), known_taps.max()))
        taps = cupy.array(_mls_taps[nbits], cupy.int64)
    else:
        taps = cupy.unique(cupy.array(taps, cupy.int64))[::-1]
        if cupy.any(taps < 0) or cupy.any(taps > nbits) or taps.size < 1:
            raise ValueError('taps must be non-empty with values between '
                             'zero and nbits (inclusive)')
        taps = cupy.array(taps)  # needed for Cython and Pythran

    n_max = (2 ** nbits) - 1
    if length is None:
        length = n_max
    else:
        length = int(length)
        if length < 0:
            raise ValueError('length must be greater than or equal to 0')

    # We use int8 instead of bool here because NumPy arrays of bools
    # don't seem to work nicely with Cython
    if state is None:
        state = cupy.ones(nbits, dtype=cupy.int8, order='c')
    else:
        # makes a copy if need be, ensuring it's 0's and 1's
        state = cupy.array(state, dtype=bool, order='c').astype(cupy.int8)
    if state.ndim != 1 or state.size != nbits:
        raise ValueError('state must be a 1-D array of size nbits')
    if cupy.all(state == 0):
        raise ValueError('state must not be all zeros')

    seq = cupy.empty(length, dtype=cupy.int8, order='c')
    n_taps = len(taps)

    _max_len_seq((1,), (nbits,), (length, n_taps, nbits, taps, state, seq))
    return seq, state
