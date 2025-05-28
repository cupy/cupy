from __future__ import annotations

import numpy

import cupy
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupy._core import _dtype


def _check_dtype(dtype: numpy.dtype | str) -> None:
    if isinstance(dtype, numpy.dtype):
        dtype = dtype.char
    if dtype not in "fdFD":
        raise RuntimeError(
            "Only float32, float64, complex64, and complex128 are supported"
        )


def _syevd(a, UPLO, with_eigen_vector, overwrite_a=False):
    from cupy_backends.cuda.libs import cublas
    from cupy_backends.cuda.libs import cusolver

    if UPLO not in ('L', 'U'):
        raise ValueError('UPLO argument must be \'L\' or \'U\'')

    # reject_float16=False for backward compatibility
    dtype, v_dtype = _util.linalg_common_type(a, reject_float16=False)
    real_dtype = dtype.char.lower()
    w_dtype = v_dtype.char.lower()

    # Note that cuSolver assumes fortran array
    v = a.astype(dtype, order='F', copy=not overwrite_a)

    m, lda = a.shape
    w = cupy.empty(m, real_dtype)
    dev_info = cupy.empty((), numpy.int32)
    handle = device.Device().cusolver_handle

    if with_eigen_vector:
        jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = cusolver.CUSOLVER_EIG_MODE_NOVECTOR

    if UPLO == 'L':
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:  # UPLO == 'U'
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if not runtime.is_hip:
        _check_dtype(dtype)
        type_v = _dtype.to_cuda_dtype(dtype)
        type_w = _dtype.to_cuda_dtype(real_dtype)
        params = cusolver.createParams()
        try:
            work_device_size, work_host_sizse = cusolver.xsyevd_bufferSize(
                handle, params, jobz, uplo, m, type_v, v.data.ptr, lda,
                type_w, w.data.ptr, type_v)
            work_device = cupy.empty(work_device_size, 'b')
            work_host = numpy.empty(work_host_sizse, 'b')
            cusolver.xsyevd(
                handle, params, jobz, uplo, m, type_v, v.data.ptr, lda,
                type_w, w.data.ptr, type_v,
                work_device.data.ptr, work_device_size,
                work_host.ctypes.data, work_host_sizse, dev_info.data.ptr)
        finally:
            cusolver.destroyParams(params)
        cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            cusolver.xsyevd, dev_info)
    else:
        if dtype == 'f':
            buffer_size = cusolver.ssyevd_bufferSize
            syevd = cusolver.ssyevd
        elif dtype == 'd':
            buffer_size = cusolver.dsyevd_bufferSize
            syevd = cusolver.dsyevd
        elif dtype == 'F':
            buffer_size = cusolver.cheevd_bufferSize
            syevd = cusolver.cheevd
        elif dtype == 'D':
            buffer_size = cusolver.zheevd_bufferSize
            syevd = cusolver.zheevd
        else:
            raise RuntimeError('Only float32, float64, complex64, and '
                               'complex128 are supported')

        work_size = buffer_size(
            handle, jobz, uplo, m, v.data.ptr, lda, w.data.ptr)
        work = cupy.empty(work_size, dtype)
        syevd(
            handle, jobz, uplo, m, v.data.ptr, lda,
            w.data.ptr, work.data.ptr, work_size, dev_info.data.ptr)
        cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            syevd, dev_info)

    return w.astype(w_dtype, copy=False), v.astype(v_dtype, copy=False)


# assemble complex eigen vectors from real eigen vectors
_assemble_complex_evs_kernel = cupy._core.ElementwiseKernel(
    'uint64 n, raw C w, raw R v_real', 'raw C v_complex',
    '''
        int col_idx = i % n;
        auto ew_i       = w[col_idx].imag();
        // if img == 0 -> ev = ev[i]
        // if img positive -> ev = ev[i] + i*ev[i+1]
        // if img negative -> ev = ev[i-1] - i*ev[i]
        int real_idx = i - ((ew_i < 0) ? 1 : 0);
        int img_idx  = i + ((ew_i > 0) ? 1 : 0);
        R factor     = ((ew_i > 0) ? R(1.0) : ((ew_i < 0) ? R(-1.0) : R(0.0)));
        v_complex[i].real(v_real[real_idx]);
        v_complex[i].imag(factor * v_real[img_idx]);
    ''',
    'cupy_assemble_complex_evs_kernel'
)


def _assemble_complex_evs(w, v_real, shape):
    n = len(w)
    v_complex = _assemble_complex_evs_kernel(n, w, v_real, size=n*n)
    return v_complex.reshape(shape)


def _geev(a, with_eigen_vector):
    from cupy_backends.cuda.libs import cusolver
    from cupyx.cusolver import check_availability
    from cupyx import empty_pinned

    if not check_availability('geev'):
        raise RuntimeError('geev is not available')
    if runtime.is_hip:
        raise NotImplementedError("geev is not implemented for HIP")

    input_dtype, _ = _util.linalg_common_type(a)
    _check_dtype(input_dtype)
    complex_dtype = numpy.dtype(input_dtype.char.upper())

    # preconvert input to be col-major for each matrix
    a = cupy.swapaxes(a, -2, -1).copy(order='C')

    if input_dtype != a.dtype:
        a_ = a.astype(input_dtype, order='C', copy=True)
    else:
        a_ = a

    m, lda = a.shape[-2:]

    w = cupy.empty(a.shape[:-1], dtype=complex_dtype)
    v = cupy.empty_like(a, dtype=complex_dtype)

    # Used for both right and (uncomputed) left eigenvectors
    real_input = input_dtype != complex_dtype
    if real_input:
        v_real = cupy.empty((m, m), dtype=input_dtype, order='F')

    dev_info = cupy.empty((), numpy.int32)
    handle = device.Device().cusolver_handle

    if with_eigen_vector:
        jobvr = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobvr = cusolver.CUSOLVER_EIG_MODE_NOVECTOR
    # Skip computing left eigenvectors
    jobvl = cusolver.CUSOLVER_EIG_MODE_NOVECTOR

    type_complex = _dtype.to_cuda_dtype(complex_dtype)
    type_input = _dtype.to_cuda_dtype(input_dtype)

    params = cusolver.createParams()
    try:
        v_ = v_real if real_input else v
        work_device_size, work_host_size = cusolver.xgeev_bufferSize(
            handle, params, jobvl, jobvr, m, type_input, a_.data.ptr, lda,
            type_complex, w.data.ptr, type_input, v_.data.ptr, lda,
            type_input, v_.data.ptr, lda, type_input)
        work_device = cupy.empty(work_device_size, 'b')
        work_host = empty_pinned(work_host_size, 'b')

        if len(a.shape) > 2:
            for ind in numpy.ndindex(a.shape[:-2]):
                a_ind = a_[ind]
                w_ind = w[ind]
                v_ind = v_real if real_input else v[ind]
                cusolver.xgeev(
                    handle, params, jobvl, jobvr, m, type_input,
                    a_ind.data.ptr, lda, type_complex, w_ind.data.ptr,
                    type_input, v_ind.data.ptr, lda, type_input,
                    v_ind.data.ptr, lda, type_input, work_device.data.ptr,
                    work_device_size, work_host.ctypes.data, work_host_size,
                    dev_info.data.ptr)
                if real_input and with_eigen_vector:
                    # in case we have real input and complex output we need to
                    # assemble complex eigen vectors from real eigen vectors
                    v[ind] = _assemble_complex_evs(w_ind, v_ind, a_ind.shape)
        else:
            cusolver.xgeev(
                handle, params, jobvl, jobvr, m, type_input, a_.data.ptr,
                lda, type_complex, w.data.ptr, type_input, v_.data.ptr, lda,
                type_input, v_.data.ptr, lda, type_input, work_device.data.ptr,
                work_device_size, work_host.ctypes.data, work_host_size,
                dev_info.data.ptr)
            if real_input and with_eigen_vector:
                # in case we have real input and complex output we need to
                # assemble complex eigen vectors from real eigen vectors
                v = _assemble_complex_evs(w, v_, a_.shape)

    finally:
        cusolver.destroyParams(params)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        cusolver.xgeev, dev_info)

    a = cupy.swapaxes(a, -2, -1).copy(order='C')

    # no need to swap axes back for real input as
    # _assemble_complex_evs already transposes
    if with_eigen_vector and not real_input:
        v = cupy.swapaxes(v, -2, -1).copy(order='C')

    return w, v


def eigh(a, UPLO='L'):
    """
    Return the eigenvalues and eigenvectors of a complex Hermitian
    (conjugate symmetric) or a real symmetric matrix.

    Returns two objects, a 1-D array containing the eigenvalues of `a`, and
    a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
    Returns:
        tuple of :class:`~cupy.ndarray`:
            Returns a tuple ``(w, v)``. ``w`` contains eigenvalues and
            ``v`` contains eigenvectors. ``v[:, i]`` is an eigenvector
            corresponding to an eigenvalue ``w[i]``. For batch input,
            ``v[k, :, i]`` is an eigenvector corresponding to an eigenvalue
            ``w[k, i]`` of ``a[k]``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.eigh`
    """
    import cupyx.cusolver
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)

    if a.size == 0:
        _, v_dtype = _util.linalg_common_type(a)
        w_dtype = v_dtype.char.lower()
        w = cupy.empty(a.shape[:-1], w_dtype)
        v = cupy.empty(a.shape, v_dtype)
        return w, v

    if a.ndim > 2 or runtime.is_hip:
        w, v = cupyx.cusolver.syevj(a, UPLO, True)
        return w, v
    else:
        return _syevd(a, UPLO, True)


def eig(a):
    """
    Return the eigenvalues and eigenvectors of a matrix.

    Returns two objects, a 1-D array containing the eigenvalues of `a`, and
    a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.
    Returns:
        tuple of :class:`~cupy.ndarray`:
            Returns a tuple ``(w, v)``. ``w`` contains eigenvalues and
            ``v`` contains eigenvectors. ``v[:, i]`` is an eigenvector
            corresponding to an eigenvalue ``w[i]``. For batch input,
            ``v[k, :, i]`` is an eigenvector corresponding to an eigenvalue
            ``w[k, i]`` of ``a[k]``.
    Notes:
        There is no guarantee of the order of the eigenvalues:
        it can even be different from ``numpy.linalg.eig``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.eig`
    """
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)

    if a.size == 0:
        _, v_dtype = _util.linalg_common_type(a)
        w = cupy.empty(a.shape[:-1], v_dtype)
        v = cupy.empty(a.shape, v_dtype)
        return w, v

    return _geev(a, True)


def eigvalsh(a, UPLO='L'):
    """
    Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

    Main difference from eigh: the eigenvectors are not computed.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
    Returns:
        cupy.ndarray:
            Returns eigenvalues as a vector ``w``. For batch input,
            ``w[k]`` is a vector of eigenvalues of matrix ``a[k]``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.eigvalsh`
    """
    import cupyx.cusolver
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)

    if a.size == 0:
        _, v_dtype = _util.linalg_common_type(a)
        w_dtype = v_dtype.char.lower()
        return cupy.empty(a.shape[:-1], w_dtype)

    if a.ndim > 2 or runtime.is_hip:
        return cupyx.cusolver.syevj(a, UPLO, False)
    else:
        return _syevd(a, UPLO, False)[0]


def eigvals(a):
    """
    Compute the eigenvalues of a matrix.

    Main difference from eig: the eigenvectors are not computed.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.
    Returns:
        cupy.ndarray:
            Returns eigenvalues as a vector ``w``. For batch input,
            ``w[k]`` is a vector of eigenvalues of matrix ``a[k]``.
    Notes:
        There is no guarantee of the order of the eigenvalues:
        it can even be different from ``numpy.linalg.eigvals``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.eigvals`
    """
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)

    if a.size == 0:
        _, v_dtype = _util.linalg_common_type(a)
        w = cupy.empty(a.shape[:-1], v_dtype)
        return w

    return _geev(a, False)[0]
