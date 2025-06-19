import unittest

import cupyx.scipy.fft as cp_fft


class TestNextFastLen(unittest.TestCase):

    def _is_fast_len(self, n):
        if n == 0:
            return True
        for p in (2, 3, 5, 7):
            while n % p == 0:
                n //= p
        return n == 1

    def test_next_fast_len(self):
        for in_value in range(2000):
            out_value = cp_fft.next_fast_len(in_value)
            assert self._is_fast_len(out_value)
            for i in range(in_value, out_value):
                assert not self._is_fast_len(i)
