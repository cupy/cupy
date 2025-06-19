"""
Routines for manipulating partial fraction expansions.
"""

import cupy


def roots(arr):
    """np.roots replacement. XXX: calls into NumPy, then converts back.
    """
    import numpy as np

    arr = cupy.asarray(arr).get()
    return cupy.asarray(np.roots(arr))


def poly(A):
    """np.poly replacement for 2D A. Otherwise, use cupy.poly."""
    sh = A.shape
    if not (len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0):
        raise ValueError("input must be a non-empty square 2d array.")

    import numpy as np

    seq_of_zeros = np.linalg.eigvals(A.get())

    dt = seq_of_zeros.dtype
    a = np.ones((1,), dtype=dt)
    for zero in seq_of_zeros:
        a = np.convolve(a, np.r_[1, -zero], mode='full')

    if issubclass(a.dtype.type, cupy.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(seq_of_zeros, dtype=complex)
        if np.all(np.sort(roots) == np.sort(roots.conjugate())):
            a = a.real.copy()

    return cupy.asarray(a)


def _cmplx_sort(p):
    """Sort roots based on magnitude.
    """
    indx = cupy.argsort(cupy.abs(p))
    return cupy.take(p, indx, 0), indx


# np.polydiv clone
def _polydiv(u, v):
    u = cupy.atleast_1d(u) + 0.0
    v = cupy.atleast_1d(v) + 0.0
    # w has the common type
    w = u[0] + v[0]
    m = len(u) - 1
    n = len(v) - 1
    scale = 1. / v[0]
    q = cupy.zeros((max(m - n + 1, 1),), w.dtype)
    r = u.astype(w.dtype)
    for k in range(0, m-n+1):
        d = scale * r[k]
        q[k] = d
        r[k:k + n + 1] -= d * v
    while cupy.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
        r = r[1:]
    return q, r


def unique_roots(p, tol=1e-3, rtype='min'):
    """Determine unique roots and their multiplicities from a list of roots.

    Parameters
    ----------
    p : array_like
        The list of roots.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. Refer to Notes about
        the details on roots grouping.
    rtype : {'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}, optional
        How to determine the returned root if multiple roots are within
        `tol` of each other.

          - 'max', 'maximum': pick the maximum of those roots
          - 'min', 'minimum': pick the minimum of those roots
          - 'avg', 'mean': take the average of those roots

        When finding minimum or maximum among complex roots they are compared
        first by the real part and then by the imaginary part.

    Returns
    -------
    unique : ndarray
        The list of unique roots.
    multiplicity : ndarray
        The multiplicity of each root.

    See Also
    --------
    scipy.signal.unique_roots

    Notes
    -----
    If we have 3 roots ``a``, ``b`` and ``c``, such that ``a`` is close to
    ``b`` and ``b`` is close to ``c`` (distance is less than `tol`), then it
    doesn't necessarily mean that ``a`` is close to ``c``. It means that roots
    grouping is not unique. In this function we use "greedy" grouping going
    through the roots in the order they are given in the input `p`.

    This utility function is not specific to roots but can be used for any
    sequence of values for which uniqueness and multiplicity has to be
    determined. For a more general routine, see `numpy.unique`.

    """
    if rtype in ['max', 'maximum']:
        reduce = cupy.max
    elif rtype in ['min', 'minimum']:
        reduce = cupy.min
    elif rtype in ['avg', 'mean']:
        reduce = cupy.mean
    else:
        raise ValueError("`rtype` must be one of "
                         "{'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}")

    points = cupy.empty((p.shape[0], 2))
    points[:, 0] = cupy.real(p)
    points[:, 1] = cupy.imag(p)

    # Replacement for dist = cdist(points, points) to avoid needing `pylibraft`
    dist = cupy.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    p_unique = []
    p_multiplicity = []
    used = cupy.zeros(p.shape[0], dtype=bool)

    for i, ds in enumerate(dist):
        if used[i]:
            continue

        mask = (ds < tol) & ~used
        group = ds[mask]
        if group.size > 0:
            # print(j, ' : ', group, p[mask])
            p_unique.append(reduce(p[mask]))
            p_multiplicity.append(group.shape[0])
        used[mask] = True

    return cupy.asarray(p_unique), cupy.asarray(p_multiplicity)


def _compute_factors(roots, multiplicity, include_powers=False):
    """Compute the total polynomial divided by factors for each root."""
    current = cupy.array([1])
    suffixes = [current]
    for pole, mult in zip(roots[-1:0:-1], multiplicity[-1:0:-1]):
        monomial = cupy.r_[1, -pole]
        for _ in range(int(mult)):
            current = cupy.polymul(current, monomial)
        suffixes.append(current)
    suffixes = suffixes[::-1]

    factors = []
    current = cupy.array([1])
    for pole, mult, suffix in zip(roots, multiplicity, suffixes):
        monomial = cupy.r_[1, -pole]
        block = []
        for i in range(int(mult)):
            if i == 0 or include_powers:
                block.append(cupy.polymul(current, suffix))
            current = cupy.polymul(current, monomial)
        factors.extend(reversed(block))

    return factors, current


def _compute_residues(poles, multiplicity, numerator):
    denominator_factors, _ = _compute_factors(poles, multiplicity)
    numerator = numerator.astype(poles.dtype)

    residues = []
    for pole, mult, factor in zip(poles, multiplicity,
                                  denominator_factors):
        if mult == 1:
            residues.append(cupy.polyval(numerator, pole) /
                            cupy.polyval(factor, pole))
        else:
            numer = numerator.copy()
            monomial = cupy.r_[1, -pole]
            factor, d = _polydiv(factor, monomial)

            block = []
            for _ in range(int(mult)):
                numer, n = _polydiv(numer, monomial)
                r = n[0] / d[0]
                numer = cupy.polysub(numer, r * factor)
                block.append(r)

            residues.extend(reversed(block))

    return cupy.asarray(residues)


def invres(r, p, k, tol=1e-3, rtype='avg'):
    """Compute b(s) and a(s) from partial fraction expansion.

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

              b(s)     b[0] s**(M) + b[1] s**(M-1) + ... + b[M]
      H(s) = ------ = ------------------------------------------
              a(s)     a[0] s**(N) + a[1] s**(N-1) + ... + a[N]

    then the partial-fraction expansion H(s) is defined as::

               r[0]       r[1]             r[-1]
           = -------- + -------- + ... + --------- + k(s)
             (s-p[0])   (s-p[1])         (s-p[-1])

    If there are any repeated roots (closer together than `tol`), then H(s)
    has terms like::

          r[i]      r[i+1]              r[i+n-1]
        -------- + ----------- + ... + -----------
        (s-p[i])  (s-p[i])**2          (s-p[i])**n

    This function is used for polynomials in positive powers of s or z,
    such as analog filters or digital filters in controls engineering.  For
    negative powers of z (typical for digital filters in DSP), use `invresz`.

    Parameters
    ----------
    r : array_like
        Residues corresponding to the poles. For repeated poles, the residues
        must be ordered to correspond to ascending by power fractions.
    p : array_like
        Poles. Equal poles must be adjacent.
    k : array_like
        Coefficients of the direct polynomial term.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    See Also
    --------
    scipy.signal.invres
    residue, invresz, unique_roots

    """
    r = cupy.atleast_1d(r)
    p = cupy.atleast_1d(p)
    k = cupy.trim_zeros(cupy.atleast_1d(k), 'f')

    unique_poles, multiplicity = unique_roots(p, tol, rtype)
    factors, denominator = _compute_factors(unique_poles, multiplicity,
                                            include_powers=True)

    if len(k) == 0:
        numerator = 0
    else:
        numerator = cupy.polymul(k, denominator)

    for residue, factor in zip(r, factors):
        numerator = cupy.polyadd(numerator, residue * factor)

    return numerator, denominator


def invresz(r, p, k, tol=1e-3, rtype='avg'):
    """Compute b(z) and a(z) from partial fraction expansion.

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

                b(z)     b[0] + b[1] z**(-1) + ... + b[M] z**(-M)
        H(z) = ------ = ------------------------------------------
                a(z)     a[0] + a[1] z**(-1) + ... + a[N] z**(-N)

    then the partial-fraction expansion H(z) is defined as::

                 r[0]                   r[-1]
         = --------------- + ... + ---------------- + k[0] + k[1]z**(-1) ...
           (1-p[0]z**(-1))         (1-p[-1]z**(-1))

    If there are any repeated roots (closer than `tol`), then the partial
    fraction expansion has terms like::

             r[i]              r[i+1]                    r[i+n-1]
        -------------- + ------------------ + ... + ------------------
        (1-p[i]z**(-1))  (1-p[i]z**(-1))**2         (1-p[i]z**(-1))**n

    This function is used for polynomials in negative powers of z,
    such as digital filters in DSP.  For positive powers, use `invres`.

    Parameters
    ----------
    r : array_like
        Residues corresponding to the poles. For repeated poles, the residues
        must be ordered to correspond to ascending by power fractions.
    p : array_like
        Poles. Equal poles must be adjacent.
    k : array_like
        Coefficients of the direct polynomial term.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    See Also
    --------
    scipy.signal.invresz
    residuez, unique_roots, invres

    """
    r = cupy.atleast_1d(r)
    p = cupy.atleast_1d(p)
    k = cupy.trim_zeros(cupy.atleast_1d(k), 'b')

    unique_poles, multiplicity = unique_roots(p, tol, rtype)
    factors, denominator = _compute_factors(unique_poles, multiplicity,
                                            include_powers=True)

    if len(k) == 0:
        numerator = 0
    else:
        numerator = cupy.polymul(k[::-1], denominator[::-1])

    for residue, factor in zip(r, factors):
        numerator = cupy.polyadd(numerator, residue * factor[::-1])

    return numerator[::-1], denominator


def residue(b, a, tol=1e-3, rtype='avg'):
    """Compute partial-fraction expansion of b(s) / a(s).

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

              b(s)     b[0] s**(M) + b[1] s**(M-1) + ... + b[M]
      H(s) = ------ = ------------------------------------------
              a(s)     a[0] s**(N) + a[1] s**(N-1) + ... + a[N]

    then the partial-fraction expansion H(s) is defined as::

               r[0]       r[1]             r[-1]
           = -------- + -------- + ... + --------- + k(s)
             (s-p[0])   (s-p[1])         (s-p[-1])

    If there are any repeated roots (closer together than `tol`), then H(s)
    has terms like::

          r[i]      r[i+1]              r[i+n-1]
        -------- + ----------- + ... + -----------
        (s-p[i])  (s-p[i])**2          (s-p[i])**n

    This function is used for polynomials in positive powers of s or z,
    such as analog filters or digital filters in controls engineering.  For
    negative powers of z (typical for digital filters in DSP), use `residuez`.

    See Notes for details about the algorithm.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    r : ndarray
        Residues corresponding to the poles. For repeated poles, the residues
        are ordered to correspond to ascending by power fractions.
    p : ndarray
        Poles ordered by magnitude in ascending order.
    k : ndarray
        Coefficients of the direct polynomial term.

    Warning
    -------
    This function may synchronize the device.

    See Also
    --------
    scipy.signal.residue
    invres, residuez, numpy.poly, unique_roots

    Notes
    -----
    The "deflation through subtraction" algorithm is used for
    computations --- method 6 in [1]_.

    The form of partial fraction expansion depends on poles multiplicity in
    the exact mathematical sense. However there is no way to exactly
    determine multiplicity of roots of a polynomial in numerical computing.
    Thus you should think of the result of `residue` with given `tol` as
    partial fraction expansion computed for the denominator composed of the
    computed poles with empirically determined multiplicity. The choice of
    `tol` can drastically change the result if there are close poles.

    References
    ----------
    .. [1] J. F. Mahoney, B. D. Sivazlian, "Partial fractions expansion: a
           review of computational methodology and efficiency", Journal of
           Computational and Applied Mathematics, Vol. 9, 1983.
    """
    if (cupy.issubdtype(b.dtype, cupy.complexfloating)
            or cupy.issubdtype(a.dtype, cupy.complexfloating)):
        b = b.astype(complex)
        a = a.astype(complex)
    else:
        b = b.astype(float)
        a = a.astype(float)

    b = cupy.trim_zeros(cupy.atleast_1d(b), 'f')
    a = cupy.trim_zeros(cupy.atleast_1d(a), 'f')

    if a.size == 0:
        raise ValueError("Denominator `a` is zero.")

    poles = roots(a)
    if b.size == 0:
        return cupy.zeros(poles.shape), _cmplx_sort(poles)[0], cupy.array([])

    if len(b) < len(a):
        k = cupy.empty(0)
    else:
        k, b = _polydiv(b, a)

    unique_poles, multiplicity = unique_roots(poles, tol=tol, rtype=rtype)
    unique_poles, order = _cmplx_sort(unique_poles)
    multiplicity = multiplicity[order]

    residues = _compute_residues(unique_poles, multiplicity, b)

    index = 0
    for pole, mult in zip(unique_poles, multiplicity):
        poles[index:index + mult] = pole
        index += mult

    return residues / a[0], poles, k


def residuez(b, a, tol=1e-3, rtype='avg'):
    """Compute partial-fraction expansion of b(z) / a(z).

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

                b(z)     b[0] + b[1] z**(-1) + ... + b[M] z**(-M)
        H(z) = ------ = ------------------------------------------
                a(z)     a[0] + a[1] z**(-1) + ... + a[N] z**(-N)

    then the partial-fraction expansion H(z) is defined as::

                 r[0]                   r[-1]
         = --------------- + ... + ---------------- + k[0] + k[1]z**(-1) ...
           (1-p[0]z**(-1))         (1-p[-1]z**(-1))

    If there are any repeated roots (closer than `tol`), then the partial
    fraction expansion has terms like::

             r[i]              r[i+1]                    r[i+n-1]
        -------------- + ------------------ + ... + ------------------
        (1-p[i]z**(-1))  (1-p[i]z**(-1))**2         (1-p[i]z**(-1))**n

    This function is used for polynomials in negative powers of z,
    such as digital filters in DSP.  For positive powers, use `residue`.

    See Notes of `residue` for details about the algorithm.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    r : ndarray
        Residues corresponding to the poles. For repeated poles, the residues
        are ordered to correspond to ascending by power fractions.
    p : ndarray
        Poles ordered by magnitude in ascending order.
    k : ndarray
        Coefficients of the direct polynomial term.

    Warning
    -------
    This function may synchronize the device.

    See Also
    --------
    scipy.signal.residuez
    invresz, residue, unique_roots
    """

    if (cupy.issubdtype(b.dtype, cupy.complexfloating)
            or cupy.issubdtype(a.dtype, cupy.complexfloating)):
        b = b.astype(complex)
        a = a.astype(complex)
    else:
        b = b.astype(float)
        a = a.astype(float)

    b = cupy.trim_zeros(cupy.atleast_1d(b), 'b')
    a = cupy.trim_zeros(cupy.atleast_1d(a), 'b')

    if a.size == 0:
        raise ValueError("Denominator `a` is zero.")
    elif a[0] == 0:
        raise ValueError("First coefficient of determinant `a` must be "
                         "non-zero.")

    poles = roots(a)
    if b.size == 0:
        return cupy.zeros(poles.shape), _cmplx_sort(poles)[0], cupy.array([])

    b_rev = b[::-1]
    a_rev = a[::-1]

    if len(b_rev) < len(a_rev):
        k_rev = cupy.empty(0)
    else:
        k_rev, b_rev = _polydiv(b_rev, a_rev)

    unique_poles, multiplicity = unique_roots(poles, tol=tol, rtype=rtype)
    unique_poles, order = _cmplx_sort(unique_poles)
    multiplicity = multiplicity[order]

    residues = _compute_residues(1 / unique_poles, multiplicity, b_rev)

    index = 0
    powers = cupy.empty(len(residues), dtype=int)
    for pole, mult in zip(unique_poles, multiplicity):
        poles[index:index + mult] = pole
        powers[index:index + mult] = 1 + cupy.arange(int(mult))
        index += mult

    residues *= (-poles) ** powers / a_rev[0]

    return residues, poles, k_rev[::-1]
