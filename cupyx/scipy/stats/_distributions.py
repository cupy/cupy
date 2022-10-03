import math

import cupy
from cupyx.scipy import special


def _normalize(x, axis):
    """Normalize, preserving floating point precision of x."""
    x_sum = x.sum(axis=axis, keepdims=True)
    if x.dtype.kind == 'f':
        x /= x_sum
    else:
        x = x / x_sum
    return x


def entropy(pk, qk=None, base=None, axis=0):
    """Calculate the entropy of a distribution for given probability values.

    If only probabilities ``pk`` are given, the entropy is calculated as
    ``S = -sum(pk * log(pk), axis=axis)``.

    If ``qk`` is not None, then compute the Kullback-Leibler divergence
    ``S = sum(pk * log(pk / qk), axis=axis)``.

    This routine will normalize ``pk`` and ``qk`` if they don't sum to 1.

    Args:
        pk (ndarray): Defines the (discrete) distribution. ``pk[i]`` is the
            (possibly unnormalized) probability of event ``i``.
        qk (ndarray, optional): Sequence against which the relative entropy is
            computed. Should be in the same format as ``pk``.
        base (float, optional): The logarithmic base to use, defaults to ``e``
            (natural logarithm).
        axis (int, optional): The axis along which the entropy is calculated.
            Default is 0.

    Returns:
        S (cupy.ndarray): The calculated entropy.

    """
    if pk.dtype.kind == 'c' or qk is not None and qk.dtype.kind == 'c':
        raise TypeError("complex dtype not supported")

    float_type = cupy.float32 if pk.dtype.char in 'ef' else cupy.float64
    pk = pk.astype(float_type, copy=False)
    pk = _normalize(pk, axis)
    if qk is None:
        vec = special.entr(pk)
    else:
        if qk.shape != pk.shape:
            raise ValueError("qk and pk must have same shape.")
        qk = qk.astype(float_type, copy=False)
        qk = _normalize(qk, axis)
        vec = special.rel_entr(pk, qk)
    s = cupy.sum(vec, axis=axis)
    if base is not None:
        s /= math.log(base)
    return s
