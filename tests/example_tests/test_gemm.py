from __future__ import annotations

from example_tests import example_test


class TestGEMM:

    def test_sgemm(self):
        example_test.run_example('gemm/sgemm.py')
