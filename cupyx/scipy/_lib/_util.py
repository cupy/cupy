import math

import cupy

from typing import (
    Optional,
    Union,
    TYPE_CHECKING,
    TypeVar,
)

IntNumber = Union[int, cupy.integer]
DecimalNumber = Union[float, cupy.floating, cupy.integer]

# Since Generator was introduced in numpy 1.17, the following condition is needed for
# backward compatibility
if TYPE_CHECKING:
    SeedType = Optional[Union[IntNumber, cupy.random.Generator,
                              cupy.random.RandomState]]
    GeneratorType = TypeVar("GeneratorType", bound=Union[cupy.random.Generator,
                                                         cupy.random.RandomState])

try:
    from cupy.random import Generator as Generator
except ImportError:
    class Generator():  # type: ignore[no-redef]
        pass
    
    
def float_factorial(n):
    """Compute the factorial and return as a float

    Returns infinity when result is too large for a double
    """
    return float(math.factorial(n)) if n < 171 else cupy.inf


def _asarray_validated(a, check_finite=True,
                       sparse_ok=False, objects_ok=False, mask_ok=False,
                       as_inexact=False):
    """Helper function for SciPy argument validation.

    Many CuPy linear algebra functions do support arbitrary array-like
    input arguments. Examples of commonly unsupported inputs include
    matrices containing inf/nan, sparse matrix representations, and
    matrices with complicated elements.

    Parameters
    ----------
    a : array-like
        The array-like input
    check_finite : bool, optional
        By default True. To check whether the input matrices contain
        only finite numbers. Disabling may give a performance gain,
        but may result in problems (crashes, non-termination) if the
        inputs do contain infinites or NaNs
    sparse_ok : bool, optional
        By default False. True if cupy sparse matrices are allowed
    objects_ok : bool, optional
        By default False. True if arrays with dype('O') are allowed
    mask_ok : bool, optional
        By default False. True if masked arrays are allowed.
    as_inexact : bool, optional
        By default False. True to convert the input array to a
        cupy.inexact dtype

    Returns
    -------
    ret : cupy.ndarray
        The converted validated array

    """

    if not sparse_ok:
        import cupyx.scipy.sparse
        if cupyx.scipy.sparse.issparse(a):
            msg = ('Sparse matrices are not supported by this function. '
                   'Perhaps one of the cupyx.scipy.sparse.linalg functions '
                   'would work instead.')
            raise ValueError(msg)

    # TODO: remove these comments when CuPy supports masked arrays
    # Ref Issue: https://github.com/cupy/cupy/issues/2225
    # if not mask_ok:
    #     if cupy.ma.isMaskedArray(a):
    #         raise ValueError('masked arrays are not supported')

    # TODO: remove these comments when CuPy supports 'object' dtype
    # if not objects_ok:
    #    if a.dtype is cupy.dtype('O'):
    #        raise ValueError('object arrays are not supported')

    if as_inexact:
        if not cupy.issubdtype(a, cupy.inexact):
            a = a.astype(dtype=cupy.float_)

    return a


def rng_integers(gen, low, high=None, size=None, dtype='int64',
                 endpoint=False):
    """
    Return random integers from low (inclusive) to high (exclusive), or if
    endpoint=True, low (inclusive) to high (inclusive). Replaces
    `RandomState.randint` (with endpoint=False) and
    `RandomState.random_integers` (with endpoint=True).

    Return random integers from the "discrete uniform" distribution of the
    specified dtype. If high is None (the default), then results are from
    0 to low.

    Parameters
    ----------
    gen : {None, np.random.RandomState, np.random.Generator}
        Random number generator. If None, then the np.random.RandomState
        singleton is used.
    low : int or array-like of ints
        Lowest (signed) integers to be drawn from the distribution (unless
        high=None, in which case this parameter is 0 and this value is used
        for high).
    high : int or array-like of ints
        If provided, one above the largest (signed) integer to be drawn from
        the distribution (see above for behavior if high=None). If array-like,
        must contain integer values.
    size : array-like of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn. Default is None, in which case a single value is
        returned.
    dtype : {str, dtype}, optional
        Desired dtype of the result. All dtypes are determined by their name,
        i.e., 'int64', 'int', etc, so byteorder is not available and a specific
        precision may have different C types depending on the platform.
        The default value is np.int_.
    endpoint : bool, optional
        If True, sample from the interval [low, high] instead of the default
        [low, high) Defaults to False.

    Returns
    -------
    out: int or ndarray of ints
        size-shaped array of random integers from the appropriate distribution,
        or a single such random int if size not provided.
    """
    if isinstance(gen, Generator):
        return gen.integers(low, high=high, size=size, dtype=dtype,
                            endpoint=endpoint)
    else:
        if gen is None:
            # default is RandomState singleton used by np.random.
            gen = cupy.random.mtrand._rand
        if endpoint:
            # inclusive of endpoint
            # remember that low and high can be arrays, so don't modify in
            # place
            if high is None:
                return gen.randint(low + 1, size=size, dtype=dtype)
            if high is not None:
                return gen.randint(low, high=high + 1, size=size, dtype=dtype)

        # exclusive
        return gen.randint(low, high=high, size=size, dtype=dtype)
    
def _rng_spawn(rng, n_children):
    # spawns independent RNGs from a parent RNG
    bg = rng._bit_generator
    ss = bg._seed_seq
    child_rngs = [cupy.random.Generator(type(bg)(child_ss))
                  for child_ss in ss.spawn(n_children)]
    return child_rngs
