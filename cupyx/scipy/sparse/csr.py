import cupy

class csr_matrix:
    def __init__(self, array):
        # create a fake CSR-like structure for demo
        self.data = cupy.array(array)
        self.shape = self.data.shape
        self.nnz = int((self.data != 0).sum())

    def __repr__(self):
        return f"<{self.shape[0]}x{self.shape[1]} sparse matrix with {self.nnz} stored elements>"
