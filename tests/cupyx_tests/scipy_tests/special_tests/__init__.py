from __future__ import annotations

import cupy
import numpy

from cupy.testing._helper import installed_but_not_baseline


def match_scipy_float32(out, xp, dtype, *, float16_only: bool = False):
    """SciPy 1.18 gave `scipy.special` float32 loops, so dtypes narrower than
    float32 return float32 where CuPy's `l->d`/`e->d` still give float64.

    Casts CuPy's result down so the values stay compared; casting SciPy's up
    would need the float64 tolerances it can no longer meet.  Does nothing
    before SciPy 1.18, nor once the baseline reaches it, so these tests then
    fail until CuPy's loops gain `e->f`/`b->f`.  Pass the dtype of the actual
    input, which is not always the swept one.  `float16_only` is for
    `polygamma`, which wraps `zeta`: its integer order keeps bool and integer
    input on the float64 loop.
    """
    if xp is not cupy or not installed_but_not_baseline(scipy="1.18"):
        return out

    dtype = numpy.dtype(dtype)
    if float16_only:
        narrowed = dtype == numpy.float16
    else:
        narrowed = (dtype != numpy.float32
                    and numpy.can_cast(dtype, numpy.float32))

    if narrowed:
        return out.astype(numpy.float32)
    return out
