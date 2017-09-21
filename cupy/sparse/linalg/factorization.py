from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device

if cuda.cusolver_enabled:
    from cupy.cuda import cusolver

def splu(A):
    ''' LU factorization for sparse matrix.
    Args:
        A (cupy.ndarray): The input matrix.
        reorder (int):
        tol (float): tolerance to decide if singular or not.

    Returns:
        singularity (int):
    '''
    if dtype == 'f':
        cusolver.scsrlsvlu()
    else:  # dtype == 'd'
        cusolver.dcsrlsvlu()
