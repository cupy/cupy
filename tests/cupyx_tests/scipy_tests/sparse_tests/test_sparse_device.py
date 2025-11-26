import cupy as cp
import cupyx.scipy.sparse as sp

# dummy csr 
def make_csr():
    data = cp.array([1, 2, 3], dtype=cp.float32)
    row = cp.array([0, 1, 2], dtype=cp.int32)
    col = cp.array([0, 1, 2], dtype=cp.int32)
    return sp.csr_matrix((data, (row, col)), shape=(3, 3))

# dummy csc
def make_csc():
    data = cp.array([1, 2, 3], dtype=cp.float32)
    row = cp.array([0, 1, 2], dtype=cp.int32)
    col = cp.array([0, 1, 2], dtype=cp.int32)
    return sp.csc_matrix((data, (row, col)), shape=(3, 3))

# dummy coo
def make_coo():
    data = cp.array([1, 2, 3], dtype=cp.float32)
    row = cp.array([0, 1, 2], dtype=cp.int32)
    col = cp.array([0, 1, 2], dtype=cp.int32)
    return sp.coo_matrix((data, (row, col)), shape=(3, 3))

# dummy dia
def make_dia():
    dia_data = cp.array([[1, 2, 3]], dtype=cp.float32)
    offsets = cp.array([0], dtype=cp.int32)
    return sp.dia_matrix((dia_data, offsets), shape=(3, 3))


def test_sparse_device():
    makers = [make_csr, make_csc, make_coo, make_dia]

    for maker in makers:
        A = maker()

        # device must exist and be a cupy.cuda.Device
        assert isinstance(A.device, cp.cuda.Device)

        # device must match the underlying data device
        assert A.device == A.data.device
