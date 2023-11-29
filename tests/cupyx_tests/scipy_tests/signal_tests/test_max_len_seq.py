
import cupy
from cupy import testing

import cupyx.scipy.signal  # NOQAs

import pytest
import numpy as np

try:
    import scipy.signal  # NOQA
except ImportError:
    pass


@testing.with_requires('scipy')
class TestMLS:

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_mls_inputs(self, mod):
        xp, scp = mod

        # can't all be zero state
        with pytest.raises(ValueError):
            scp.signal.max_len_seq(10, state=xp.zeros(10))

        # wrong size state
        with pytest.raises(ValueError):
            scp.signal.max_len_seq(10, state=xp.ones(3))

        # wrong length
        with pytest.raises(ValueError):
            scp.signal.max_len_seq(10, length=-1)

        assert scp.signal.max_len_seq(10, length=0)[0].size == 0

        # unknown taps
        with pytest.raises(ValueError):
            scp.signal.max_len_seq(64)

        # bad taps
        with pytest.raises(ValueError):
            scp.signal.max_len_seq(10, taps=[-1, 1])

    @pytest.mark.parametrize('nbits', list(range(2, 8)))
    @pytest.mark.parametrize('state', [None, 'rand'])
    @pytest.mark.parametrize('taps', [None, 'custom'])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_mls_output(self, nbits, state, taps, xp, scp):
        # define some alternate working taps
        alt_taps = {2: [1], 3: [2], 4: [3], 5: [4, 3, 2], 6: [5, 4, 1], 7: [4],
                    8: [7, 5, 3]}
        # assume the other bit levels work, too slow to test higher orders...
        if state == 'rand':
            state = xp.round(testing.shaped_random((nbits,), xp, scale=1))

        if taps == 'custom':
            taps = alt_taps[nbits]

        if state is not None and xp.all(state == 0):
            state[0] = 1  # they can't all be zero

        results = []
        orig_m = scp.signal.max_len_seq(nbits, state=state, taps=taps)[0]
        m = 2. * orig_m - 1.  # convert to +/- 1 representation
        results.append(m)
        results.append(orig_m)

        # Test via circular cross-correlation, which is just mult.
        # in the frequency domain with one signal conjugated
        tester = xp.real(
            scp.fft.ifft(scp.fft.fft(m) * xp.conj(scp.fft.fft(m))))
        out_len = 2 ** nbits - 1

        # impulse amplitude == test_len
        # steady-state is -1
        results.append(tester)

        # let's do the split thing using a couple options
        for n in (1, 2 ** (nbits - 1)):
            m1, s1 = scp.signal.max_len_seq(
                nbits, state=state, taps=taps, length=n)
            m2, s2 = scp.signal.max_len_seq(
                nbits, state=s1, taps=taps, length=1)
            m3, _ = scp.signal.max_len_seq(
                nbits, state=s2, taps=taps, length=out_len - n - 1)
            new_m = xp.concatenate((m1, m2, m3))
            results.append(new_m)

        return results
