#
# Example of matrix multiply using cuSPARSELt
#
# (*) https://docs.nvidia.com/cuda/cusparselt/getting_started.html#code-example
#
import cupy
import numpy

from cupy.cuda import runtime
from cupy_backends.cuda.libs.cusparselt import Handle, MatDescriptor, MatmulDescriptor, MatmulAlgSelection, MatmulPlan  # NOQA
from cupy_backends.cuda.libs import cusparselt, cusparse

dtype = 'float16'
m, n, k = 1024, 1024, 1024
A = cupy.random.random((m, k)).astype(dtype)
B = cupy.ones((k, n), dtype=dtype)
C = cupy.zeros((m, n), dtype=dtype)

#
# initializes cusparselt handle and data structures
#
handle = Handle()
matA = MatDescriptor()
matB = MatDescriptor()
matC = MatDescriptor()
matmul = MatmulDescriptor()
alg_sel = MatmulAlgSelection()
plan = MatmulPlan()
cusparselt.init(handle)

#
# initializes matrix descriptors
#
alignment = 128
order = cusparse.CUSPARSE_ORDER_ROW
cuda_dtype = runtime.CUDA_R_16F
cusparselt.structuredDescriptorInit(handle, matA, A.shape[0], A.shape[1],
                                    A.shape[1], alignment, cuda_dtype, order,
                                    cusparselt.CUSPARSELT_SPARSITY_50_PERCENT)
cusparselt.denseDescriptorInit(handle, matB, B.shape[0], B.shape[1],
                               B.shape[1], alignment, cuda_dtype, order)
cusparselt.denseDescriptorInit(handle, matC, C.shape[0], C.shape[1],
                               C.shape[1], alignment, cuda_dtype, order)

#
# initializes matmul, algorithm selection and plan
#
opA = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
opB = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
compute_type = cusparselt.CUSPARSE_COMPUTE_16F
cusparselt.matmulDescriptorInit(handle, matmul, opA, opB, matA, matB, matC,
                                matC, compute_type)
cusparselt.matmulAlgSelectionInit(handle, alg_sel, matmul,
                                  cusparselt.CUSPARSELT_MATMUL_ALG_DEFAULT)
alg = numpy.array(0, dtype='int32')
cusparselt.matmulAlgSetAttribute(handle, alg_sel,
                                 cusparselt.CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                 alg.ctypes.data, 4)
workspace_size = cusparselt.matmulGetWorkspace(handle, alg_sel)
workspace = cupy.empty(workspace_size, dtype='int8')
cusparselt.matmulPlanInit(handle, plan, matmul, alg_sel, workspace_size)

#
# prunes the matrix A in-place and checks the correstness
#
print('Before pruning, A[0]:\n{}'.format(A[0]))
cusparselt.spMMAPrune(handle, matmul, A.data.ptr, A.data.ptr,
                      cusparselt.CUSPARSELT_PRUNE_SPMMA_TILE)
print('After pruning, A[0]:\n{}'.format(A[0]))
is_valid = numpy.array(-1, dtype='int32')
cusparselt.spMMAPruneCheck(handle, matmul, A.data.ptr, is_valid.ctypes.data)

#
# compresses the matrix A
#
compressed_size = cusparselt.spMMACompressedSize(handle, plan)
A_compressed = cupy.zeros(compressed_size, dtype='uint8')
cusparselt.spMMACompress(handle, plan, A.data.ptr, A_compressed.data.ptr)

#
# matmul: C = A @ B
#
alpha = numpy.array(1.0, dtype='float32')
beta = numpy.array(0.0, dtype='float32')
cusparselt.matmul(handle, plan, alpha.ctypes.data, A_compressed.data.ptr,
                  B.data.ptr, beta.ctypes.data, C.data.ptr, C.data.ptr,
                  workspace.data.ptr)

print('A.sum(axis=1): {}'.format(A.sum(axis=1)))
print('C[:, 0]: {}'.format(C[:, 0]))

#
# destroys plan and handle
#
cusparselt.matDescriptorDestroy(matA)
cusparselt.matDescriptorDestroy(matB)
cusparselt.matDescriptorDestroy(matC)
cusparselt.matmulPlanDestroy(plan)
cusparselt.destroy(handle)
