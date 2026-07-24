from __future__ import annotations

import cupy as cp
from cupy import testing
import cupyx.scipy.sparse as sp


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc', 'coo', 'dia'],
}))
class TestSparseDevice:

    def test_sparse_device(self):
        data = cp.array([1, 2, 3], dtype=cp.float32)

        if self.format == 'dia':
            A = sp.dia_matrix(
                (data[None, :], cp.array([0], dtype=cp.int32)),
                shape=(3, 3)
            )
        else:
            idx = cp.array([0, 1, 2], dtype=cp.int32)
            mat_cls = getattr(sp, f'{self.format}_matrix')
            A = mat_cls((data, (idx, idx)), shape=(3, 3))

        assert isinstance(A.device, cp.cuda.Device)
        assert A.device == A.data.device
