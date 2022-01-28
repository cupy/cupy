import numpy
import cupy
import cupy.linalg as linalg
# waiting implementation of the following modules in PR #4172
# from cupyx.scipy.linalg import (cho_factor, cho_solve)
from cupyx.scipy.sparse import linalg as splinalg


def _cholesky(B):
    """
    Wrapper around `cupy.linalg.cholesky` that raises LinAlgError if there are
    NaNs in the output
    """
    R = cupy.linalg.cholesky(B)
    if cupy.any(cupy.isnan(R)):
        raise numpy.linalg.LinAlgError
    return R


# TODO: This helper function can be replaced after cupy.block is supported
def _bmat(list_obj):
    """
    Helper function to create a block matrix in cupy from a list
    of smaller 2D dense arrays
    """
    n_rows = len(list_obj)
    n_cols = len(list_obj[0])
    final_shape = [0, 0]
    # calculating expected size of output
    for i in range(n_rows):
        final_shape[0] += list_obj[i][0].shape[0]
    for j in range(n_cols):
        final_shape[1] += list_obj[0][j].shape[1]
    # obtaining result's datatype
    dtype = cupy.result_type(*[arr.dtype for
                               list_iter in list_obj for arr in list_iter])
    # checking order
    F_order = all(arr.flags['F_CONTIGUOUS'] for list_iter
                  in list_obj for arr in list_iter)
    C_order = all(arr.flags['C_CONTIGUOUS'] for list_iter
                  in list_obj for arr in list_iter)
    order = 'F' if F_order and not C_order else 'C'
    result = cupy.empty(tuple(final_shape), dtype=dtype, order=order)

    start_idx_row = 0
    start_idx_col = 0
    end_idx_row = 0
    end_idx_col = 0
    for i in range(n_rows):
        end_idx_row = start_idx_row + list_obj[i][0].shape[0]
        start_idx_col = 0
        for j in range(n_cols):
            end_idx_col = start_idx_col + list_obj[i][j].shape[1]
            result[start_idx_row:end_idx_row,
                   start_idx_col: end_idx_col] = list_obj[i][j]
            start_idx_col = end_idx_col
        start_idx_row = end_idx_row
    return result


def _report_nonhermitian(M, name):
    """
    Report if `M` is not a hermitian matrix given its type.
    """

    md = M - M.T.conj()

    nmd = linalg.norm(md, 1)
    tol = 10 * cupy.finfo(M.dtype).eps
    tol *= max(1, float(linalg.norm(M, 1)))
    if nmd > tol:
        print('matrix %s of the type %s is not sufficiently Hermitian:'
              % (name, M.dtype))
        print('condition: %.e < %e' % (nmd, tol))


def _as2d(ar):
    """
    If the input array is 2D return it, if it is 1D, append a dimension,
    making it a column vector.
    """
    if ar.ndim == 2:
        return ar
    else:  # Assume 1!
        aux = cupy.array(ar, copy=False)
        aux.shape = (ar.shape[0], 1)
        return aux


def _makeOperator(operatorInput, expectedShape):
    """Takes a dense numpy array or a sparse matrix or
    a function and makes an operator performing matrix * blockvector
    products.
    """
    if operatorInput is None:
        return None
    else:
        operator = splinalg.aslinearoperator(operatorInput)

    if operator.shape != expectedShape:
        raise ValueError('operator has invalid shape')

    return operator


def _applyConstraints(blockVectorV, YBY, blockVectorBY, blockVectorY):
    """Changes blockVectorV in place."""
    YBV = cupy.dot(blockVectorBY.T.conj(), blockVectorV)
    # awaiting the implementation of cho_solve in PR #4172
    # tmp = cho_solve(factYBY, YBV)
    tmp = linalg.solve(YBY, YBV)
    blockVectorV -= cupy.dot(blockVectorY, tmp)


def _b_orthonormalize(B, blockVectorV, blockVectorBV=None, retInvR=False):
    """B-orthonormalize the given block vector using Cholesky."""
    normalization = blockVectorV.max(
        axis=0) + cupy.finfo(blockVectorV.dtype).eps
    blockVectorV = blockVectorV / normalization
    if blockVectorBV is None:
        if B is not None:
            blockVectorBV = B(blockVectorV)
        else:
            blockVectorBV = blockVectorV
    else:
        blockVectorBV = blockVectorBV / normalization
    VBV = cupy.matmul(blockVectorV.T.conj(), blockVectorBV)
    try:
        # VBV is a Cholesky factor
        VBV = _cholesky(VBV)
        VBV = linalg.inv(VBV.T)
        blockVectorV = cupy.matmul(blockVectorV, VBV)
        if B is not None:
            blockVectorBV = cupy.matmul(blockVectorBV, VBV)
        else:
            blockVectorBV = None
    except numpy.linalg.LinAlgError:
        # LinAlg Error: cholesky transformation might fail in rare cases
        # raise ValueError("cholesky has failed")
        blockVectorV = None
        blockVectorBV = None
        VBV = None

    if retInvR:
        return blockVectorV, blockVectorBV, VBV, normalization
    else:
        return blockVectorV, blockVectorBV


def _get_indx(_lambda, num, largest):
    """Get `num` indices into `_lambda` depending on `largest` option."""
    ii = cupy.argsort(_lambda)
    if largest:
        ii = ii[:-num - 1:-1]
    else:
        ii = ii[:num]
    return ii


# TODO: This helper function can be replaced after cupy.eigh
#       supports generalized eigen value problems.
def _eigh(A, B=None):
    """
    Helper function for converting a generalized eigenvalue problem
    A(X) = lambda(B(X)) to standard eigen value problem using cholesky
    transformation
    """
    if(B is None):  # use cupy's eigh in standard case
        vals, vecs = linalg.eigh(A)
        return vals, vecs
    R = _cholesky(B)
    RTi = linalg.inv(R)
    Ri = linalg.inv(R.T)
    F = cupy.matmul(RTi, cupy.matmul(A, Ri))
    vals, vecs = linalg.eigh(F)
    eigVec = cupy.matmul(Ri, vecs)
    return vals, eigVec


def lobpcg(A, X,
           B=None, M=None, Y=None,
           tol=None, maxiter=None,
           largest=True, verbosityLevel=0,
           retLambdaHistory=False, retResidualNormsHistory=False):
    """Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)

    LOBPCG is a preconditioned eigensolver for large symmetric positive
    definite (SPD) generalized eigenproblems.

    Args:
        A (array-like): The symmetric linear operator of the problem,
            usually a sparse matrix. Can be of the following types
            - cupy.ndarray
            - cupyx.scipy.sparse.csr_matrix
            - cupy.scipy.sparse.linalg.LinearOperator
        X (cupy.ndarray): Initial approximation to the ``k``
            eigenvectors (non-sparse). If `A` has ``shape=(n,n)``
            then `X` should have shape ``shape=(n,k)``.
        B (array-like): The right hand side operator in a generalized
            eigenproblem. By default, ``B = Identity``.
            Can be of following types:
            - cupy.ndarray
            - cupyx.scipy.sparse.csr_matrix
            - cupy.scipy.sparse.linalg.LinearOperator
        M (array-like): Preconditioner to `A`; by default ``M = Identity``.
            `M` should approximate the inverse of `A`.
            Can be of the following types:
            - cupy.ndarray
            - cupyx.scipy.sparse.csr_matrix
            - cupy.scipy.sparse.linalg.LinearOperator
        Y (cupy.ndarray):
            `n-by-sizeY` matrix of constraints (non-sparse), `sizeY < n`
            The iterations will be performed in the B-orthogonal complement
            of the column-space of Y. Y must be full rank.
        tol (float):
            Solver tolerance (stopping criterion).
            The default is ``tol=n*sqrt(eps)``.
        maxiter (int):
            Maximum number of iterations.  The default is ``maxiter = 20``.
        largest (bool):
            When True, solve for the largest eigenvalues,
            otherwise the smallest.
        verbosityLevel (int):
            Controls solver output.  The default is ``verbosityLevel=0``.
        retLambdaHistory (bool):
            Whether to return eigenvalue history.  Default is False.
        retResidualNormsHistory (bool):
            Whether to return history of residual norms.  Default is False.

    Returns:
        tuple:
            - `w` (cupy.ndarray): Array of ``k`` eigenvalues
            - `v` (cupy.ndarray) An array of ``k`` eigenvectors.
              `v` has the same shape as `X`.
            - `lambdas` (list of cupy.ndarray): The eigenvalue history,
              if `retLambdaHistory` is True.
            - `rnorms` (list of cupy.ndarray): The history of residual norms,
              if `retResidualNormsHistory` is True.

    .. seealso:: :func:`scipy.sparse.linalg.lobpcg`

    .. note::
        If both ``retLambdaHistory`` and ``retResidualNormsHistory`` are `True`
        the return tuple has the following format
        ``(lambda, V, lambda history, residual norms history)``.
    """
    blockVectorX = X
    blockVectorY = Y
    residualTolerance = tol

    if maxiter is None:
        maxiter = 20

    if blockVectorY is not None:
        sizeY = blockVectorY.shape[1]
    else:
        sizeY = 0

    if len(blockVectorX.shape) != 2:
        raise ValueError('expected rank-2 array for argument X')

    n, sizeX = blockVectorX.shape

    if verbosityLevel:
        aux = "Solving "
        if B is None:
            aux += "standard"
        else:
            aux += "generalized"
        aux += " eigenvalue problem with"
        if M is None:
            aux += "out"
        aux += " preconditioning\n\n"
        aux += "matrix size %d\n" % n
        aux += "block size %d\n\n" % sizeX
        if blockVectorY is None:
            aux += "No constraints\n\n"
        else:
            if sizeY > 1:
                aux += "%d constraints\n\n" % sizeY
            else:
                aux += "%d constraint\n\n" % sizeY
        print(aux)

    A = _makeOperator(A, (n, n))
    B = _makeOperator(B, (n, n))
    M = _makeOperator(M, (n, n))

    if (n - sizeY) < (5 * sizeX):
        # The problem size is small compared to the block size.
        # Using dense general eigensolver instead of LOBPCG.
        sizeX = min(sizeX, n)

        if blockVectorY is not None:
            raise NotImplementedError('The dense eigensolver '
                                      'does not support constraints.')

        A_dense = A(cupy.eye(n, dtype=A.dtype))
        B_dense = None if B is None else B(cupy.eye(n, dtype=B.dtype))

        # call numerically unstable general eigen solver
        vals, vecs = _eigh(A_dense, B_dense)
        if largest:
            # Reverse order to be compatible with eigs() in 'LM' mode.
            vals = vals[::-1]
            vecs = vecs[:, ::-1]

        vals = vals[:sizeX]
        vecs = vecs[:, :sizeX]

        return vals, vecs

    if (residualTolerance is None) or (residualTolerance <= 0.0):
        residualTolerance = cupy.sqrt(1e-15) * n

    # Apply constraints to X.
    if blockVectorY is not None:

        if B is not None:
            blockVectorBY = B(blockVectorY)
        else:
            blockVectorBY = blockVectorY

        # gramYBY is a dense array.
        gramYBY = cupy.dot(blockVectorY.T.conj(), blockVectorBY)

        # awaiting implementation of cho_factor in PR #4172
        # try:
        #    gramYBY is a Cholesky factor from now on...
        #    gramYBY = cho_factor(gramYBY)
        # except numpy.linalg.LinAlgError:
        #    raise ValueError("cannot handle linearly dependent constraints")

        _applyConstraints(blockVectorX, gramYBY, blockVectorBY, blockVectorY)

    # B-orthonormalize X.
    blockVectorX, blockVectorBX = _b_orthonormalize(B, blockVectorX)

    # Compute the initial Ritz vectors: solve the eigenproblem.
    blockVectorAX = A(blockVectorX)
    gramXAX = cupy.dot(blockVectorX.T.conj(), blockVectorAX)

    _lambda, eigBlockVector = _eigh(gramXAX)
    ii = _get_indx(_lambda, sizeX, largest)
    _lambda = _lambda[ii]

    eigBlockVector = cupy.asarray(eigBlockVector[:, ii])
    blockVectorX = cupy.dot(blockVectorX, eigBlockVector)
    blockVectorAX = cupy.dot(blockVectorAX, eigBlockVector)
    if B is not None:
        blockVectorBX = cupy.dot(blockVectorBX, eigBlockVector)

    # Active index set.
    activeMask = cupy.ones((sizeX,), dtype=bool)

    lambdaHistory = [_lambda]
    residualNormsHistory = []

    previousBlockSize = sizeX
    ident = cupy.eye(sizeX, dtype=A.dtype)
    ident0 = cupy.eye(sizeX, dtype=A.dtype)

    ##
    # Main iteration loop.

    blockVectorP = None  # set during iteration
    blockVectorAP = None
    blockVectorBP = None

    iterationNumber = -1
    restart = True
    explicitGramFlag = False
    while iterationNumber < maxiter:
        iterationNumber += 1
        if verbosityLevel > 0:
            print('iteration %d' % iterationNumber)

        if B is not None:
            aux = blockVectorBX * _lambda[cupy.newaxis, :]
        else:
            aux = blockVectorX * _lambda[cupy.newaxis, :]

        blockVectorR = blockVectorAX - aux

        aux = cupy.sum(blockVectorR.conj() * blockVectorR, 0)
        residualNorms = cupy.sqrt(aux)

        residualNormsHistory.append(residualNorms)

        ii = cupy.where(residualNorms > residualTolerance, True, False)
        activeMask = activeMask & ii
        if verbosityLevel > 2:
            print(activeMask)

        currentBlockSize = int(activeMask.sum())
        if currentBlockSize != previousBlockSize:
            previousBlockSize = currentBlockSize
            ident = cupy.eye(currentBlockSize, dtype=A.dtype)

        if currentBlockSize == 0:
            break

        if verbosityLevel > 0:
            print('current block size:', currentBlockSize)
            print('eigenvalue:', _lambda)
            print('residual norms:', residualNorms)
        if verbosityLevel > 10:
            print(eigBlockVector)

        activeBlockVectorR = _as2d(blockVectorR[:, activeMask])

        if iterationNumber > 0:
            activeBlockVectorP = _as2d(blockVectorP[:, activeMask])
            activeBlockVectorAP = _as2d(blockVectorAP[:, activeMask])
            if B is not None:
                activeBlockVectorBP = _as2d(blockVectorBP[:, activeMask])

        if M is not None:
            # Apply preconditioner T to the active residuals.
            activeBlockVectorR = M(activeBlockVectorR)

        # Apply constraints to the preconditioned residuals.
        if blockVectorY is not None:
            _applyConstraints(activeBlockVectorR,
                              gramYBY, blockVectorBY, blockVectorY)

        # B-orthogonalize the preconditioned residuals to X.
        if B is not None:
            activeBlockVectorR = activeBlockVectorR\
                - cupy.matmul(blockVectorX,
                              cupy
                              .matmul(blockVectorBX.T.conj(),
                                      activeBlockVectorR))
        else:
            activeBlockVectorR = activeBlockVectorR - \
                cupy.matmul(blockVectorX,
                            cupy.matmul(blockVectorX.T.conj(),
                                        activeBlockVectorR))

        ##
        # B-orthonormalize the preconditioned residuals.
        aux = _b_orthonormalize(B, activeBlockVectorR)
        activeBlockVectorR, activeBlockVectorBR = aux

        activeBlockVectorAR = A(activeBlockVectorR)

        if iterationNumber > 0:
            if B is not None:
                aux = _b_orthonormalize(B, activeBlockVectorP,
                                        activeBlockVectorBP, retInvR=True)
                activeBlockVectorP, activeBlockVectorBP, invR, normal = aux
            else:
                aux = _b_orthonormalize(B, activeBlockVectorP, retInvR=True)
                activeBlockVectorP, _, invR, normal = aux
            # Function _b_orthonormalize returns None if Cholesky fails
            if activeBlockVectorP is not None:
                activeBlockVectorAP = activeBlockVectorAP / normal
                activeBlockVectorAP = cupy.dot(activeBlockVectorAP, invR)
                restart = False
            else:
                restart = True

        ##
        # Perform the Rayleigh Ritz Procedure:
        # Compute symmetric Gram matrices:

        if activeBlockVectorAR.dtype == 'float32':
            myeps = 1
        elif activeBlockVectorR.dtype == 'float32':
            myeps = 1e-4
        else:
            myeps = 1e-8

        if residualNorms.max() > myeps and not explicitGramFlag:
            explicitGramFlag = False
        else:
            # Once explicitGramFlag, forever explicitGramFlag.
            explicitGramFlag = True

        # Shared memory assingments to simplify the code
        if B is None:
            blockVectorBX = blockVectorX
            activeBlockVectorBR = activeBlockVectorR
            if not restart:
                activeBlockVectorBP = activeBlockVectorP

        # Common submatrices:
        gramXAR = cupy.dot(blockVectorX.T.conj(), activeBlockVectorAR)
        gramRAR = cupy.dot(activeBlockVectorR.T.conj(), activeBlockVectorAR)

        if explicitGramFlag:
            gramRAR = (gramRAR + gramRAR.T.conj()) / 2
            gramXAX = cupy.dot(blockVectorX.T.conj(), blockVectorAX)
            gramXAX = (gramXAX + gramXAX.T.conj()) / 2
            gramXBX = cupy.dot(blockVectorX.T.conj(), blockVectorBX)
            gramRBR = cupy.dot(activeBlockVectorR.T.conj(),
                               activeBlockVectorBR)
            gramXBR = cupy.dot(blockVectorX.T.conj(), activeBlockVectorBR)
        else:
            gramXAX = cupy.diag(_lambda)
            gramXBX = ident0
            gramRBR = ident
            gramXBR = cupy.zeros((int(sizeX), int(currentBlockSize)),
                                 dtype=A.dtype)

        def _handle_gramA_gramB_verbosity(gramA, gramB):
            if verbosityLevel > 0:
                _report_nonhermitian(gramA, 'gramA')
                _report_nonhermitian(gramB, 'gramB')
            if verbosityLevel > 10:
                # Note: not documented, but leave it in here for now
                numpy.savetxt('gramA.txt', cupy.asnumpy(gramA))
                numpy.savetxt('gramB.txt', cupy.asnumpy(gramB))

        if not restart:
            gramXAP = cupy.dot(blockVectorX.T.conj(), activeBlockVectorAP)
            gramRAP = cupy.dot(activeBlockVectorR.T.conj(),
                               activeBlockVectorAP)
            gramPAP = cupy.dot(activeBlockVectorP.T.conj(),
                               activeBlockVectorAP)
            gramXBP = cupy.dot(blockVectorX.T.conj(), activeBlockVectorBP)
            gramRBP = cupy.dot(activeBlockVectorR.T.conj(),
                               activeBlockVectorBP)
            if explicitGramFlag:
                gramPAP = (gramPAP + gramPAP.T.conj()) / 2
                gramPBP = cupy.dot(activeBlockVectorP.T.conj(),
                                   activeBlockVectorBP)
            else:
                gramPBP = ident

            gramA = _bmat([[gramXAX, gramXAR, gramXAP],
                           [gramXAR.T.conj(), gramRAR, gramRAP],
                           [gramXAP.T.conj(), gramRAP.T.conj(), gramPAP]])
            gramB = _bmat([[gramXBX, gramXBR, gramXBP],
                           [gramXBR.T.conj(), gramRBR, gramRBP],
                           [gramXBP.T.conj(), gramRBP.T.conj(), gramPBP]])

            _handle_gramA_gramB_verbosity(gramA, gramB)

            try:
                _lambda, eigBlockVector = _eigh(gramA, gramB)
            except numpy.linalg.LinAlgError:
                # try again after dropping the direction vectors P from RR
                restart = True

        if restart:
            gramA = _bmat([[gramXAX, gramXAR],
                           [gramXAR.T.conj(), gramRAR]])
            gramB = _bmat([[gramXBX, gramXBR],
                           [gramXBR.T.conj(), gramRBR]])

            _handle_gramA_gramB_verbosity(gramA, gramB)

            try:
                _lambda, eigBlockVector = _eigh(gramA, gramB)
            except numpy.linalg.LinAlgError:
                raise ValueError('eigh has failed in lobpcg iterations')

        ii = _get_indx(_lambda, sizeX, largest)
        if verbosityLevel > 10:
            print(ii)
            print(_lambda)

        _lambda = _lambda[ii]
        eigBlockVector = eigBlockVector[:, ii]

        lambdaHistory.append(_lambda)

        if verbosityLevel > 10:
            print('lambda:', _lambda)

        if verbosityLevel > 10:
            print(eigBlockVector)

        # Compute Ritz vectors.
        if B is not None:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:sizeX +
                                                 currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]

                pp = cupy.dot(activeBlockVectorR, eigBlockVectorR)
                pp += cupy.dot(activeBlockVectorP, eigBlockVectorP)

                app = cupy.dot(activeBlockVectorAR, eigBlockVectorR)
                app += cupy.dot(activeBlockVectorAP, eigBlockVectorP)

                bpp = cupy.dot(activeBlockVectorBR, eigBlockVectorR)
                bpp += cupy.dot(activeBlockVectorBP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]

                pp = cupy.dot(activeBlockVectorR, eigBlockVectorR)
                app = cupy.dot(activeBlockVectorAR, eigBlockVectorR)
                bpp = cupy.dot(activeBlockVectorBR, eigBlockVectorR)

            if verbosityLevel > 10:
                print(pp)
                print(app)
                print(bpp)

            blockVectorX = cupy.dot(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = cupy.dot(blockVectorAX, eigBlockVectorX) + app
            blockVectorBX = cupy.dot(blockVectorBX, eigBlockVectorX) + bpp

            blockVectorP, blockVectorAP, blockVectorBP = pp, app, bpp

        else:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:sizeX +
                                                 currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]

                pp = cupy.dot(activeBlockVectorR, eigBlockVectorR)
                pp += cupy.dot(activeBlockVectorP, eigBlockVectorP)

                app = cupy.dot(activeBlockVectorAR, eigBlockVectorR)
                app += cupy.dot(activeBlockVectorAP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]

                pp = cupy.dot(activeBlockVectorR, eigBlockVectorR)
                app = cupy.dot(activeBlockVectorAR, eigBlockVectorR)

            if verbosityLevel > 10:
                print(pp)
                print(app)

            blockVectorX = cupy.dot(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = cupy.dot(blockVectorAX, eigBlockVectorX) + app

            blockVectorP, blockVectorAP = pp, app

    if B is not None:
        aux = blockVectorBX * _lambda[cupy.newaxis, :]

    else:
        aux = blockVectorX * _lambda[cupy.newaxis, :]

    blockVectorR = blockVectorAX - aux

    aux = cupy.sum(blockVectorR.conj() * blockVectorR, 0)
    residualNorms = cupy.sqrt(aux)

    # Future work:
    # Generalized eigen value solver like `scipy.linalg.eigh`
    # that takes in `B` matrix as input
    # `cupy.linalg.cholesky` is more unstable than `scipy.linalg.cholesky`
    # Making sure eigenvectors "exactly" satisfy the blockVectorY constrains?
    # Making sure eigenvecotrs are "exactly" othonormalized by final "exact" RR
    # Computing the actual true residuals

    if verbosityLevel > 0:
        print('final eigenvalue:', _lambda)
        print('final residual norms:', residualNorms)

    if retLambdaHistory:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, lambdaHistory, residualNormsHistory
        else:
            return _lambda, blockVectorX, lambdaHistory
    else:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, residualNormsHistory
        else:
            return _lambda, blockVectorX
