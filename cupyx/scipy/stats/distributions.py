import math
import numpy

import cupy
from cupyx.scipy import special


def entropy(pk, qk=None, base=None, axis=0):
    """Calculate the entropy of a distribution for given probability values.

    If only probabilities `pk` are given, the entropy is calculated as
    ``S = -sum(pk * log(pk), axis=axis)``.

    If `qk` is not None, then compute the Kullback-Leibler divergence
    ``S = sum(pk * log(pk / qk), axis=axis)``.

    This routine will normalize `pk` and `qk` if they don't sum to 1.

    Args:
        pk (sequence): Defines the (discrete) distribution. ``pk[i]`` is the
            (possibly unnormalized) probability of event ``i``.
        qk (sequence, optional): Sequence against which the relative entropy is
            computed. Should be in the same format as `pk`.
        base (float, optional): The logarithmic base to use, defaults to ``e``
            (natural logarithm).
        axis (int, optional): The axis along which the entropy is calculated.
            Default is 0.

    Returns:
        S (cupy.ndarray): The calculated entropy.

    """
    pk = pk.astype(cupy.promote_types(pk.dtype, cupy.float32))
    pk_sum = cupy.sum(pk, axis=axis, keepdims=True)
    if pk.dtype.kind == 'f':
        pk /= pk_sum
    else:
        pk = pk / pk_sum
    if qk is None:
        vec = special.entr(pk)
    else:
        if qk.shape != pk.shape:
            raise ValueError("qk and pk must have same shape.")
        qk = qk.astype(cupy.promote_types(qk.dtype, cupy.float32))
        qk_sum = cupy.sum(qk, axis=axis, keepdims=True)
        if qk.dtype.kind == 'f':
            qk /= qk_sum
        else:
            qk = qk / qk_sum
        vec = special.rel_entr(pk, qk)
    s = cupy.sum(vec, axis=axis)
    if base is not None:
        if s.ndim == 0:
            # required to match scipy output dtype
            s = s.astype(numpy.float64)
        s /= math.log(base)
    return s
