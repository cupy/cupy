from __future__ import annotations

import threading
from unittest import mock

import numpy
import pytest

import cupy
from cupy._core import _cuda_compute_common
from cupy._core import _cuda_compute_merge_sort as merge_sort
from cupy._core import _accelerator
from cupy.cuda._compiler_cache import DiskKernelCacheBackend
from cupy.cuda import thrust


@pytest.fixture(autouse=True)
def require_cuda_compute():
    if _cuda_compute_common._get_cuda_compute() is None:
        pytest.skip('cuda.compute (cuda-cccl) not found')


def _expected(host, descending):
    if descending:
        if host.dtype.kind in 'fc':
            # Negate before and after a stable sort to reverse numeric order
            # while retaining NaN category order and the input order of ties
            # (including signed zeros and distinct NaN payloads).
            return -numpy.sort(-host, axis=-1, kind='stable')
        # Bitwise inversion reverses integer/bool order without overflow,
        # preserving stability rather than reversing equal-element groups.
        return ~numpy.sort(~host, axis=-1, kind='stable')
    return numpy.sort(host, axis=-1, kind='stable')


def _offset_allocator(offset):
    allocator = cupy.cuda.get_allocator()

    def allocate(size):
        return allocator(size + offset) + offset

    return allocate


@pytest.mark.parametrize('offset', range(8))
@pytest.mark.parametrize('size', [0, 3])
def test_key_buffer_alignment(offset, size):
    dtype = merge_sort._key_dtype(numpy.dtype('complex128'))
    with cupy.cuda.using_allocator(_offset_allocator(offset)):
        keys = merge_sort._KeyBuffer(size, dtype)
    assert isinstance(keys.data.ptr, int)
    assert keys.data.ptr % dtype.alignment == 0
    assert keys.__cuda_array_interface__['data'][0] == keys.data.ptr


@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
@pytest.mark.parametrize('descending', [False, True])
def test_sort_with_misaligned_key_storage(dtype, descending):
    host = numpy.arange(102, dtype=numpy.float64)[::-1].astype(dtype)
    host = host.reshape(3, 34)
    a = cupy.asarray(host)
    key_buffer = merge_sort._KeyBuffer
    allocator = _offset_allocator(1)

    def allocate_keys(size, dtype):
        with cupy.cuda.using_allocator(allocator):
            return key_buffer(size, dtype)

    with mock.patch.object(
            merge_sort, '_KeyBuffer', side_effect=allocate_keys):
        result = merge_sort.cuda_compute_merge_sort(a, descending)
    assert result is not a
    numpy.testing.assert_array_equal(
        cupy.asnumpy(result), _expected(host, descending))
    numpy.testing.assert_array_equal(cupy.asnumpy(a), host)


@pytest.mark.parametrize('dtype', sorted(merge_sort._SUPPORTED_DTYPES))
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('multi_row', [False, True])
def test_sort(dtype, descending, multi_row):
    rng = numpy.random.default_rng(42)
    shape = (2, 3, 7513) if multi_row else (7513,)
    host = rng.integers(0, 100, size=shape).astype(dtype)
    if host.dtype.kind == 'b':
        host = rng.integers(0, 2, size=shape).astype(dtype)
    elif host.dtype.kind == 'f':
        host -= 50
        host.flat[:8] = [numpy.nan, -numpy.inf, numpy.inf, -0., 0.,
                         numpy.nan, -1., 1.]
    elif host.dtype.kind == 'c':
        host += (1j * rng.integers(-10, 11, size=shape)).astype(dtype)
        components = [-numpy.inf, -1., -0., 0., 2., numpy.inf, numpy.nan]
        special = [complex(r, i) for r in components for i in components]
        host.flat[:len(special)] = special
    a = cupy.asarray(host)
    ready = cupy.cuda.get_current_stream().record()
    with cupy.cuda.Stream(non_blocking=True) as stream:
        stream.wait_event(ready)
        result = merge_sort.cuda_compute_merge_sort(a, descending)
        assert result is not a
    stream.synchronize()
    actual = cupy.asnumpy(result)
    numpy.testing.assert_array_equal(cupy.asnumpy(a), host)
    expected = _expected(host, descending)
    if host.dtype.kind == 'c':
        numpy.testing.assert_array_equal(actual.real, expected.real)
        numpy.testing.assert_array_equal(actual.imag, expected.imag)
    else:
        numpy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('multi_row', [False, True])
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('dtype', [
    'float16', 'float32', 'float64', 'complex64', 'complex128'])
def test_stability(multi_row, descending, dtype):
    # Signed zeros and distinct NaN payloads compare equivalent. Preserve their
    # exact input order across merge tiles, rather than only equality.
    dtype = numpy.dtype(dtype)
    component_dtype = numpy.empty((), dtype=dtype).real.dtype
    patterns = {
        2: [0x8000, 0, 0x7e01, 0x7e02, 0x3c00],
        4: [0x80000000, 0, 0x7fc00001, 0x7fc00002, 0x3f800000],
        8: [0x8000000000000000, 0, 0x7ff8000000000001,
            0x7ff8000000000002, 0x3ff0000000000000],
    }
    bits = numpy.array(
        patterns[component_dtype.itemsize],
        dtype=f'uint{component_dtype.itemsize * 8}')
    components = numpy.tile(bits.view(component_dtype), 2101)
    if dtype.kind == 'c':
        host = numpy.empty(components.shape, dtype=dtype)
        host.real = components
        host.imag = numpy.roll(components, 1)
    else:
        host = components
    if multi_row:
        host = numpy.stack([host, host[::-1]])
    a = cupy.asarray(host)
    result = merge_sort.cuda_compute_merge_sort(a, descending)
    numpy.testing.assert_array_equal(
        cupy.asnumpy(result).view(numpy.uint8),
        _expected(host, descending).view(numpy.uint8))


@pytest.mark.parametrize('shape', [(0,), (3, 0), (0, 4), (1,), (2, 1)])
def test_empty_and_singleton(shape):
    a = cupy.empty(shape, dtype=numpy.float32)
    with mock.patch.object(merge_sort, '_run_merge_sort') as build:
        result = merge_sort.cuda_compute_merge_sort(a)
        assert result is not a
        assert result.shape == a.shape
        assert result.dtype == a.dtype
        numpy.testing.assert_array_equal(cupy.asnumpy(result), cupy.asnumpy(a))
    build.assert_not_called()


def test_unsupported():
    assert merge_sort.cuda_compute_merge_sort(numpy.ones(3)) is None
    assert merge_sort.cuda_compute_merge_sort(cupy.ones(())) is None
    assert merge_sort.cuda_compute_merge_sort(cupy.ones(10)[::2]) is None


def test_unavailable():
    with mock.patch.object(merge_sort, '_get_cuda_compute', return_value=None):
        assert merge_sort.cuda_compute_merge_sort(cupy.ones(3)) is None


@pytest.mark.parametrize('shape', [(3,), (1, 3)])
def test_allocation_failure_leaves_input_unchanged(shape):
    host = numpy.array([3, 1, 2], dtype=numpy.int32).reshape(shape)
    a = cupy.asarray(host)
    # Warm compilation and caches before injecting execution-time failures.
    result = merge_sort.cuda_compute_merge_sort(a)
    assert result is not None
    cupy.cuda.get_current_stream().synchronize()
    allocator = cupy.cuda.get_allocator()

    # Fail each allocation in turn, stopping when the operation completes
    # without reaching the failure point. No allocation count or order is
    # prescribed, so changing the implementation does not require an update.
    failure_index = 0
    while True:
        allocation_count = 0
        failure_injected = False

        def allocate(size):
            nonlocal allocation_count, failure_injected
            index = allocation_count
            allocation_count += 1
            if index == failure_index:
                failure_injected = True
                raise cupy.cuda.memory.OutOfMemoryError(size, 0)
            return allocator(size)

        with cupy.cuda.using_allocator(allocate):
            result = merge_sort.cuda_compute_merge_sort(a)
        numpy.testing.assert_array_equal(cupy.asnumpy(a), host)
        if not failure_injected:
            assert failure_index > 0
            assert result is not None
            numpy.testing.assert_array_equal(
                cupy.asnumpy(result), numpy.sort(host, axis=-1))
            break
        assert result is None
        failure_index += 1


@pytest.mark.thread_unsafe(reason='Patches global cache and factories.')
@pytest.mark.parametrize('operation', ['sort', 'argsort', 'lexsort'])
def test_cache_round_trip_and_shape_reuse(tmp_path, operation):
    compute = _cuda_compute_common._get_cuda_compute()
    backend = DiskKernelCacheBackend(str(tmp_path))
    errors = []

    def run():
        try:
            # Same ndim family and dtype: different row widths must reuse the
            # algorithm, since width is runtime state. 1-D has its own family.
            shapes = [(17,), (29,), (2, 17), (3, 29)]
            if operation == 'lexsort':
                shapes = shapes[2:]
            for shape in shapes:
                host = numpy.arange(numpy.prod(shape), dtype=numpy.int32)
                host = host[::-1].copy().reshape(shape)
                a = cupy.asarray(host)
                if operation == 'sort':
                    result = merge_sort.cuda_compute_merge_sort(a)
                    expected = numpy.sort(host, axis=-1)
                elif operation == 'argsort':
                    result = merge_sort.cuda_compute_argsort(a)
                    expected = numpy.argsort(host, axis=-1, kind='stable')
                else:
                    result = merge_sort.cuda_compute_lexsort(a)
                    expected = numpy.lexsort(host)
                numpy.testing.assert_array_equal(
                    cupy.asnumpy(result), expected)
        except BaseException as error:
            errors.append(error)

    def in_new_thread():
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        if errors:
            raise errors[0]

    with mock.patch('cupy.cuda.compiler._kernel_cache_backend', backend):
        with mock.patch.object(
                compute, 'make_merge_sort',
                wraps=compute.make_merge_sort) as build:
            in_new_thread()
            assert build.call_count == (1 if operation == 'lexsort' else 2)
        family = 'merge_sort' if operation == 'sort' else f'merge_{operation}'
        if operation == 'lexsort':
            assert len(list(tmp_path.glob(f'*.cc_{family}'))) == 1
        else:
            assert len(list(tmp_path.glob(f'*.cc_{family}_1d'))) == 1
            assert len(list(tmp_path.glob(f'*.cc_{family}_nd'))) == 1
        with mock.patch.object(
                compute, 'make_merge_sort',
                side_effect=AssertionError('Expected a disk-cache hit')):
            in_new_thread()


@pytest.mark.parametrize('dtype', sorted(merge_sort._SUPPORTED_DTYPES))
def test_argsort(dtype):
    rng = numpy.random.default_rng(42)
    host = rng.integers(0, 10, size=(2, 7513)).astype(dtype)
    if host.dtype.kind == 'b':
        host = (host == 0)
    elif host.dtype.kind in 'fc':
        host[:, :4] = [numpy.nan, -0., 0., numpy.nan]
        if host.dtype.kind == 'c':
            host[:, 4:8] = [complex(numpy.nan, 1), complex(1, numpy.nan),
                            complex(numpy.nan, numpy.nan), complex(1, -1)]
    for multi_row in [False, True]:
        data = host if multi_row else host[0]
        a = cupy.asarray(data)
        for descending in [False, True]:
            order_values = data
            if descending:
                if data.dtype.kind in 'biu':
                    order_values = data.astype(numpy.int64)
                order_values = -order_values
            result = merge_sort.cuda_compute_argsort(a, descending)
            assert result.dtype == numpy.dtype(numpy.intp)
            numpy.testing.assert_array_equal(
                cupy.asnumpy(result),
                numpy.argsort(order_values, axis=-1, kind='stable'))
        numpy.testing.assert_array_equal(cupy.asnumpy(a), data)


@pytest.mark.parametrize('dtype', sorted(merge_sort._SUPPORTED_DTYPES))
def test_lexsort(dtype):
    rng = numpy.random.default_rng(43)
    host = rng.integers(0, 5, size=(3, 7513)).astype(dtype)
    if host.dtype.kind == 'b':
        host = (host == 0)
    elif host.dtype.kind in 'fc':
        host[:, :4] = [numpy.nan, -0., 0., numpy.nan]
        if host.dtype.kind == 'c':
            host[:, 4:8] = [complex(numpy.nan, 1), complex(1, numpy.nan),
                            complex(numpy.nan, numpy.nan), complex(1, -1)]
    a = cupy.asarray(host)
    result = merge_sort.cuda_compute_lexsort(a)
    assert result.dtype == numpy.dtype(numpy.intp)
    numpy.testing.assert_array_equal(cupy.asnumpy(result), numpy.lexsort(host))
    numpy.testing.assert_array_equal(cupy.asnumpy(a), host)


def test_indirect_sort_empty_and_singleton():
    for shape in [(0,), (2, 0), (0, 4), (1,), (2, 1)]:
        a = cupy.zeros(shape)
        result = merge_sort.cuda_compute_argsort(a)
        numpy.testing.assert_array_equal(
            cupy.asnumpy(result),
            numpy.argsort(cupy.asnumpy(a), kind='stable'))
    for shape in [(2, 0), (2, 1), (0, 5)]:
        a = cupy.zeros(shape)
        result = merge_sort.cuda_compute_lexsort(a)
        numpy.testing.assert_array_equal(
            cupy.asnumpy(result), numpy.arange(shape[-1]))


@pytest.mark.thread_unsafe(reason='Changes routine accelerators and backends.')
class TestSortDispatch:
    @pytest.fixture(autouse=True)
    def enable_cuda_compute(self):
        previous = _accelerator.get_routine_accelerators()[:]
        _accelerator.set_routine_accelerators(['cuda_compute'])
        try:
            yield
        finally:
            _accelerator.set_routine_accelerators(previous)

    @pytest.mark.parametrize('dtype', ['int32', 'float32', 'complex128'])
    @pytest.mark.parametrize('axis', [-1, 0, 1])
    @pytest.mark.parametrize('descending', [False, True])
    @pytest.mark.parametrize('use_method', [False, True])
    def test_dispatch(self, dtype, axis, descending, use_method):
        host = numpy.arange(2 * 3 * 257)[::-1].copy().reshape(2, 3, 257)
        host = host.astype(dtype)
        if host.dtype.kind in 'fc':
            host.flat[0] = numpy.nan
        a = cupy.asarray(host)
        with mock.patch.object(
                merge_sort, 'cuda_compute_merge_sort',
                wraps=merge_sort.cuda_compute_merge_sort) as accelerated:
            with mock.patch.object(thrust, 'available', False):
                if use_method:
                    assert a.sort(axis=axis, descending=descending) is None
                    result = a
                else:
                    result = cupy.sort(
                        a, axis=axis, descending=descending)
                    numpy.testing.assert_array_equal(cupy.asnumpy(a), host)
        accelerated.assert_called_once()
        backend_input = accelerated.call_args.args[0]
        assert backend_input.flags.c_contiguous
        assert backend_input.shape[-1] == host.shape[axis]
        expected = numpy.moveaxis(
            _expected(numpy.moveaxis(host, axis, -1), descending), -1, axis)
        numpy.testing.assert_array_equal(cupy.asnumpy(result), expected)

    @pytest.mark.parametrize('axis', [None, -1, 0])
    def test_noncontiguous_input(self, axis):
        host = numpy.arange(60)[::-1].reshape(5, 12)[:, ::2]
        a = cupy.asarray(host).T
        host = host.T
        with mock.patch.object(thrust, 'available', False):
            result = cupy.sort(a, axis=axis)
        numpy.testing.assert_array_equal(
            cupy.asnumpy(result), numpy.sort(host, axis=axis))
        numpy.testing.assert_array_equal(cupy.asnumpy(a), host)

    @pytest.mark.parametrize('dtype', ['float32', 'complex128'])
    @pytest.mark.parametrize('use_method', [False, True])
    def test_argsort_dispatch(self, dtype, use_method):
        rng = numpy.random.default_rng(44)
        host = rng.integers(0, 5, (3, 37)).astype(dtype)
        host[0, :3] = [numpy.nan, 0., -0.]
        a = cupy.asarray(host).T
        host = host.T
        for axis in [-1, 0, None]:
            for descending in [False, True]:
                with mock.patch.object(
                        merge_sort, 'cuda_compute_argsort',
                        wraps=merge_sort.cuda_compute_argsort) as accelerated:
                    with mock.patch.object(thrust, 'available', False):
                        if use_method:
                            result = a.argsort(
                                axis=axis, descending=descending)
                        else:
                            result = cupy.argsort(
                                a, axis=axis, descending=descending)
                accelerated.assert_called_once()
                assert accelerated.call_args.args[0].flags.c_contiguous
                expected = numpy.argsort(
                    -host if descending else host, axis=axis, kind='stable')
                numpy.testing.assert_array_equal(
                    cupy.asnumpy(result), expected)
        numpy.testing.assert_array_equal(cupy.asnumpy(a), host)

    @pytest.mark.parametrize('dtype', ['float32', 'complex128'])
    @pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
    def test_lexsort_dispatch(self, dtype, layout):
        rng = numpy.random.default_rng(45)
        host = rng.integers(0, 5, (3, 79)).astype(dtype)
        host[:, :3] = [numpy.nan, 0., -0.]
        a = cupy.array(host, order='F' if layout == 'F' else 'C')
        if layout == 'strided':
            a = a[:, ::-2]
            host = host[:, ::-2]
        with mock.patch.object(
                merge_sort, 'cuda_compute_lexsort',
                wraps=merge_sort.cuda_compute_lexsort) as accelerated:
            with mock.patch.object(thrust, 'available', False):
                result = cupy.lexsort(a)
        accelerated.assert_called_once()
        assert accelerated.call_args.args[0].flags.c_contiguous
        numpy.testing.assert_array_equal(
            cupy.asnumpy(result), numpy.lexsort(host))
        numpy.testing.assert_array_equal(cupy.asnumpy(a), host)

    @pytest.mark.parametrize('operation', ['sort', 'argsort', 'lexsort'])
    @pytest.mark.parametrize(
        'reason', ['disabled', 'declined', 'missing', 'oom'])
    def test_thrust_fallback(self, reason, operation):
        if not thrust.available:
            pytest.skip('Thrust is unavailable')
        if reason == 'disabled':
            _accelerator.set_routine_accelerators([])
        builder = '_run_merge_sort'
        entry = ('cuda_compute_merge_sort' if operation == 'sort'
                 else f'cuda_compute_{operation}')
        patches = {
            'disabled': mock.patch.object(merge_sort, builder),
            'declined': mock.patch.object(
                merge_sort, entry, return_value=None),
            'missing': mock.patch.object(
                merge_sort, '_get_cuda_compute', return_value=None),
            'oom': mock.patch.object(
                merge_sort, builder,
                side_effect=cupy.cuda.memory.OutOfMemoryError(1024, 0)),
        }
        a = cupy.asarray([[6, 4, 5], [3, 1, 2]], dtype=numpy.int32)
        host = cupy.asnumpy(a)
        with patches[reason] as patched:
            with mock.patch.object(
                    thrust, operation,
                    wraps=getattr(thrust, operation)) as fallback:
                if operation == 'sort':
                    a.sort(axis=0)
                    result = a
                    expected = numpy.sort(host, axis=0)
                elif operation == 'argsort':
                    result = a.argsort(axis=0)
                    expected = numpy.argsort(host, axis=0, kind='stable')
                else:
                    result = cupy.lexsort(a)
                    expected = numpy.lexsort(host)
        fallback.assert_called_once()
        if reason == 'disabled':
            patched.assert_not_called()
        else:
            patched.assert_called_once()
        numpy.testing.assert_array_equal(
            cupy.asnumpy(result), expected)
