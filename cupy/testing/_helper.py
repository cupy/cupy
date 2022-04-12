import contextlib
import inspect
from typing import Callable
import unittest
from unittest import mock
import warnings

import numpy

import cupy
from cupy._core import internal
import cupyx
import cupyx.scipy.sparse

from cupy.testing._pytest_impl import is_available


if is_available():
    import pytest
    _skipif: Callable[..., Callable[[Callable], Callable]] = pytest.mark.skipif
else:
    _skipif = unittest.skipIf


def with_requires(*requirements):
    """Run a test case only when given requirements are satisfied.

    .. admonition:: Example

       This test case runs only when `numpy>=1.18` is installed.

       >>> from cupy import testing
       ... class Test(unittest.TestCase):
       ...     @testing.with_requires('numpy>=1.18')
       ...     def test_for_numpy_1_18(self):
       ...         pass

    Args:
        requirements: A list of string representing requirement condition to
            run a given test case.

    """
    msg = 'requires: {}'.format(','.join(requirements))
    return _skipif(not installed(requirements), reason=msg)


def installed(*specifiers):
    """Returns True if the current environment satisfies the specified
    package requirement.

    Args:
        specifiers: Version specifiers (e.g., `numpy>=1.20.0`).
    """
    # Delay import of pkg_resources because it is excruciatingly slow.
    # See https://github.com/pypa/setuptools/issues/510
    import pkg_resources

    for spec in specifiers:
        try:
            pkg_resources.require(spec)
        except pkg_resources.ResolutionError:
            return False
    return True


def numpy_satisfies(version_range):
    """Returns True if numpy version satisfies the specified criteria.

    Args:
        version_range: A version specifier (e.g., `>=1.13.0`).
    """
    return installed('numpy{}'.format(version_range))


def shaped_arange(shape, xp=cupy, dtype=numpy.float32, order='C'):
    """Returns an array with given shape, array module, and dtype.

    Args:
         shape(tuple of int): Shape of returned ndarray.
         xp(numpy or cupy): Array module to use.
         dtype(dtype): Dtype of returned ndarray.
         order({'C', 'F'}): Order of returned ndarray.

    Returns:
         numpy.ndarray or cupy.ndarray:
         The array filled with :math:`1, \\cdots, N` with specified dtype
         with given shape, array module. Here, :math:`N` is
         the size of the returned array.
         If ``dtype`` is ``numpy.bool_``, evens (resp. odds) are converted to
         ``True`` (resp. ``False``).

    """
    dtype = numpy.dtype(dtype)
    a = numpy.arange(1, internal.prod(shape) + 1, 1)
    if dtype == '?':
        a = a % 2 == 0
    elif dtype.kind == 'c':
        a = a + a * 1j
    return xp.array(a.astype(dtype).reshape(shape), order=order)


def shaped_reverse_arange(shape, xp=cupy, dtype=numpy.float32):
    """Returns an array filled with decreasing numbers.

    Args:
         shape(tuple of int): Shape of returned ndarray.
         xp(numpy or cupy): Array module to use.
         dtype(dtype): Dtype of returned ndarray.

    Returns:
         numpy.ndarray or cupy.ndarray:
         The array filled with :math:`N, \\cdots, 1` with specified dtype
         with given shape, array module.
         Here, :math:`N` is the size of the returned array.
         If ``dtype`` is ``numpy.bool_``, evens (resp. odds) are converted to
         ``True`` (resp. ``False``).
    """
    dtype = numpy.dtype(dtype)
    size = internal.prod(shape)
    a = numpy.arange(size, 0, -1)
    if dtype == '?':
        a = a % 2 == 0
    elif dtype.kind == 'c':
        a = a + a * 1j
    return xp.array(a.astype(dtype).reshape(shape))


def shaped_random(
        shape, xp=cupy, dtype=numpy.float32, scale=10, seed=0, order='C'):
    """Returns an array filled with random values.

    Args:
         shape(tuple): Shape of returned ndarray.
         xp(numpy or cupy): Array module to use.
         dtype(dtype): Dtype of returned ndarray.
         scale(float): Scaling factor of elements.
         seed(int): Random seed.

    Returns:
         numpy.ndarray or cupy.ndarray: The array with
         given shape, array module,

    If ``dtype`` is ``numpy.bool_``, the elements are
    independently drawn from ``True`` and ``False``
    with same probabilities.
    Otherwise, the array is filled with samples
    independently and identically drawn
    from uniform distribution over :math:`[0, scale)`
    with specified dtype.
    """
    numpy.random.seed(seed)
    dtype = numpy.dtype(dtype)
    if dtype == '?':
        a = numpy.random.randint(2, size=shape)
    elif dtype.kind == 'c':
        a = numpy.random.rand(*shape) + 1j * numpy.random.rand(*shape)
        a *= scale
    else:
        a = numpy.random.rand(*shape) * scale
    return xp.asarray(a, dtype=dtype, order=order)


def shaped_sparse_random(
        shape, sp=cupyx.scipy.sparse, dtype=numpy.float32,
        density=0.01, format='coo', seed=0):
    """Returns an array filled with random values.

    Args:
        shape (tuple): Shape of returned sparse matrix.
        sp (scipy.sparse or cupyx.scipy.sparse): Sparce matrix module to use.
        dtype (dtype): Dtype of returned sparse matrix.
        density (float): Density of returned sparse matrix.
        format (str): Format of returned sparse matrix.
        seed (int): Random seed.

    Returns:
        The sparse matrix with given shape, array module,
    """
    import scipy.sparse
    n_rows, n_cols = shape
    numpy.random.seed(seed)
    a = scipy.sparse.random(n_rows, n_cols, density).astype(dtype)

    if sp is cupyx.scipy.sparse:
        a = cupyx.scipy.sparse.coo_matrix(a)
    elif sp is not scipy.sparse:
        raise ValueError('Unknown module: {}'.format(sp))

    return a.asformat(format)


def generate_matrix(
        shape, xp=cupy, dtype=numpy.float32, *, singular_values=None):
    r"""Returns a matrix with specified singular values.

    Generates a random matrix with given singular values.
    This function generates a random NumPy matrix (or a stack of matrices) that
    has specified singular values. It can be used to generate the inputs for a
    test that can be instable when the input value behaves bad.
    Notation: denote the shape of the generated array by :math:`(B..., M, N)`,
    and :math:`K = min\{M, N\}`. :math:`B...` may be an empty sequence.

    Args:
        shape (tuple of int): Shape of the generated array, i.e.,
            :math:`(B..., M, N)`.
        xp (numpy or cupy): Array module to use.
        dtype: Dtype of the generated array.
        singular_values (array-like): Singular values of the generated
            matrices. It must be broadcastable to shape :math:`(B..., K)`.

    Returns:
        numpy.ndarray or cupy.ndarray: A random matrix that has specifiec
        singular values.
    """

    if len(shape) <= 1:
        raise ValueError(
            'shape {} is invalid for matrices: too few axes'.format(shape)
        )

    if singular_values is None:
        raise TypeError('singular_values is not given')
    singular_values = xp.asarray(singular_values)

    dtype = numpy.dtype(dtype)
    if dtype.kind not in 'fc':
        raise TypeError('dtype {} is not supported'.format(dtype))

    if not xp.isrealobj(singular_values):
        raise TypeError('singular_values is not real')
    if (singular_values < 0).any():
        raise ValueError('negative singular value is given')

    # Generate random matrices with given singular values. We simply generate
    # orthogonal vectors using SVD on random matrices and then combine them
    # with the given singular values.
    a = xp.random.randn(*shape)
    if dtype.kind == 'c':
        a = a + 1j * xp.random.randn(*shape)
    u, s, vh = xp.linalg.svd(a, full_matrices=False)
    sv = xp.broadcast_to(singular_values, s.shape)
    a = xp.einsum('...ik,...k,...kj->...ij', u, sv, vh)
    return a.astype(dtype)


@contextlib.contextmanager
def assert_warns(expected):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        yield

    if any(isinstance(m.message, expected) for m in w):
        return

    try:
        exc_name = expected.__name__
    except AttributeError:
        exc_name = str(expected)

    raise AssertionError('%s not triggerred' % exc_name)


class NumpyAliasTestBase(unittest.TestCase):

    @property
    def func(self):
        raise NotImplementedError()

    @property
    def cupy_func(self):
        return getattr(cupy, self.func)

    @property
    def numpy_func(self):
        return getattr(numpy, self.func)


class NumpyAliasBasicTestBase(NumpyAliasTestBase):

    def test_argspec(self):
        f = inspect.signature
        assert f(self.cupy_func) == f(self.numpy_func)

    def test_docstring(self):
        cupy_func = self.cupy_func
        numpy_func = self.numpy_func
        assert hasattr(cupy_func, '__doc__')
        assert cupy_func.__doc__ is not None
        assert cupy_func.__doc__ != ''
        assert cupy_func.__doc__ is not numpy_func.__doc__


class NumpyAliasValuesTestBase(NumpyAliasTestBase):

    def test_values(self):
        assert self.cupy_func(*self.args) == self.numpy_func(*self.args)


@contextlib.contextmanager
def assert_function_is_called(*args, times_called=1, **kwargs):
    """A handy wrapper for unittest.mock to check if a function is called.

    Args:
        *args: Arguments of `mock.patch`.
        times_called (int): The number of times the function should be
            called. Default is ``1``.
        **kwargs: Keyword arguments of `mock.patch`.

    """
    with mock.patch(*args, **kwargs) as handle:
        yield
        assert handle.call_count == times_called


# TODO(kataoka): remove this alias
AssertFunctionIsCalled = assert_function_is_called
