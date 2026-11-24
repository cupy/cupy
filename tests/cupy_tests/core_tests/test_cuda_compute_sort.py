from __future__ import annotations

import threading
from unittest import mock

import numpy
import pytest

import cupy
from cupy._core import _cuda_compute_common
from cupy._core import _cuda_compute_sort as compute_sort
from cupy._core import _accelerator
from cupy.cuda._compiler_cache import DiskKernelCacheBackend
from cupy.cuda import thrust


@pytest.fixture(autouse=True)
def require_cuda_compute():
    if _cuda_compute_common._get_cuda_compute() is None:
        pytest.skip('cuda.compute (cuda-cccl) not found')


def _expected(host, descending):
    if host.dtype.name == 'bfloat16':
        keys = host.astype(numpy.float32)
        indices = numpy.argsort(
            -keys if descending else keys, axis=-1, kind='stable')
        return numpy.take_along_axis(host, indices, axis=-1)
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
    dtype = compute_sort._key_dtype(numpy.dtype('complex128'))
    with cupy.cuda.using_allocator(_offset_allocator(offset)):
        keys = compute_sort._KeyBuffer(size, dtype)
    assert isinstance(keys.data.ptr, int)
    assert keys.data.ptr % dtype.alignment == 0
    assert keys.__cuda_array_interface__['data'][0] == keys.data.ptr


@pytest.mark.thread_unsafe(reason='Patches the global key-buffer factory.')
@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
@pytest.mark.parametrize('descending', [False, True])
def test_sort_with_misaligned_key_storage(dtype, descending):
    host = numpy.arange(102, dtype=numpy.float64)[::-1].astype(dtype)
    host = host.reshape(3, 34)
    a = cupy.asarray(host)
    key_buffer = compute_sort._KeyBuffer
    allocator = _offset_allocator(1)

    def allocate_keys(size, dtype):
        with cupy.cuda.using_allocator(allocator):
            return key_buffer(size, dtype)

    with mock.patch.object(
            compute_sort, '_KeyBuffer', side_effect=allocate_keys):
        result = compute_sort.cuda_compute_sort(a, descending)
    assert result is not a
    numpy.testing.assert_array_equal(
        cupy.asnumpy(result), _expected(host, descending))
    numpy.testing.assert_array_equal(cupy.asnumpy(a), host)


@pytest.mark.parametrize('algorithm,shape', [
    ('merge', (4,)), ('merge', (10007,)),
    ('merge', (2, 10007)), ('merge', (2, 3, 257)),
    ('radix_transform', (4,)), ('radix_transform', (10007,)),
    ('auto', (32767,)), ('auto', (32768,)), ('auto', (32769,)),
    ('auto', (2, 10007)),
])
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('return_indices', [False, True])
def test_bool_storage(algorithm, shape, descending, return_indices):
    # Every nonzero byte is logically true. Compare canonical logical values
    # for the reference: NumPy's bool radix sort may order the raw bytes.
    patterns = numpy.concatenate((
        numpy.arange(256, dtype=numpy.uint8), numpy.zeros(256, numpy.uint8)))
    raw = (numpy.array([2, 1, 255, 0], dtype=numpy.uint8) if shape == (4,)
           else numpy.resize(patterns, numpy.prod(shape)).reshape(shape))
    if shape != (4,):
        numpy.random.default_rng(53).shuffle(raw, axis=-1)
    logical = raw != 0
    indices = numpy.argsort(
        ~logical if descending else logical, axis=-1, kind='stable')
    sort = (compute_sort.cuda_compute_argsort if return_indices
            else compute_sort.cuda_compute_sort)
    with cupy.cuda.Stream(non_blocking=True):
        a = cupy.asarray(raw).view(cupy.bool_)
        result = sort(a, descending, algorithm=algorithm)
        assert result.dtype == numpy.dtype(numpy.intp if return_indices
                                           else numpy.bool_)
        actual = cupy.asnumpy(result if return_indices
                              else result.view(cupy.uint8))
        numpy.testing.assert_array_equal(cupy.asnumpy(a.view(cupy.uint8)), raw)
    expected = (indices if return_indices
                else numpy.take_along_axis(raw, indices, axis=-1))
    numpy.testing.assert_array_equal(actual, expected)


def test_bool_storage_lexsort():
    raw = numpy.resize(numpy.arange(256, dtype=numpy.uint8), (3, 10007))
    numpy.random.default_rng(54).shuffle(raw, axis=-1)
    a = cupy.asarray(raw).view(cupy.bool_)
    result = compute_sort.cuda_compute_lexsort(a)
    numpy.testing.assert_array_equal(
        cupy.asnumpy(result), numpy.lexsort(raw != 0))
    numpy.testing.assert_array_equal(cupy.asnumpy(a.view(cupy.uint8)), raw)


@pytest.mark.parametrize('size', [4, 10007])
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('return_indices', [False, True])
@pytest.mark.parametrize('overwrite_input', [False, True])
def test_native_bool_radix(size, descending, return_indices, overwrite_input):
    # Native ordering distinguishes true representations; it must consider
    # all eight bits, rather than the single bit used by normalized keys.
    raw = numpy.resize(numpy.array(
        [2, 1, 255, 0], dtype=numpy.uint8), size)
    if size > 4:
        raw = numpy.resize(numpy.arange(256, dtype=numpy.uint8), size)
        numpy.random.default_rng(55).shuffle(raw)
    indices = numpy.argsort(~raw if descending else raw, kind='stable')
    sort = (compute_sort.cuda_compute_argsort if return_indices
            else compute_sort.cuda_compute_sort)
    with cupy.cuda.Stream(non_blocking=True):
        a = cupy.asarray(raw).view(cupy.bool_)
        result = sort(a, descending, algorithm='radix_native',
                      overwrite_input=overwrite_input)
        assert result.dtype == numpy.dtype(numpy.intp if return_indices
                                           else numpy.bool_)
        actual = cupy.asnumpy(result if return_indices
                              else result.view(cupy.uint8))
        if not overwrite_input:
            numpy.testing.assert_array_equal(
                cupy.asnumpy(a.view(cupy.uint8)), raw)
        elif not return_indices:
            assert result is a
    numpy.testing.assert_array_equal(
        actual, indices if return_indices else raw[indices])


@pytest.mark.parametrize('dtype', [
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'])
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('return_indices', [False, True])
def test_integer_1d(dtype, descending, return_indices):
    limits = numpy.iinfo(dtype)
    # Include both limits and repeated keys to exercise signed radix ordering,
    # full-width unsigned keys, and stable argsort in both directions.
    values = numpy.array(
        [limits.min, limits.max, 0, 1, limits.max, limits.min, 1, 0],
        dtype=dtype)
    host = numpy.resize(values, 10007)
    numpy.random.default_rng(46).shuffle(host)
    a = cupy.asarray(host)
    if return_indices:
        result = compute_sort.cuda_compute_argsort(a, descending)
        expected = numpy.argsort(
            ~host if descending else host, kind='stable')
        assert result.dtype == numpy.dtype(numpy.intp)
    else:
        result = compute_sort.cuda_compute_sort(a, descending)
        expected = _expected(host, descending)
    numpy.testing.assert_array_equal(cupy.asnumpy(result), expected)
    numpy.testing.assert_array_equal(cupy.asnumpy(a), host)


@pytest.mark.parametrize('dtype,algorithm', [
    ('int16', 'auto'), ('int32', 'auto'),
    ('float32', 'radix_native'), ('float32', 'radix_transform'),
    ('bool', 'radix_transform'),
] + ([('bfloat16', 'radix_transform')]
     if cupy._util.BF16 is not None else []))
@pytest.mark.parametrize('size', [2048, 10007])
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('return_indices', [False, True])
def test_radix_writable_input(
        dtype, algorithm, size, descending, return_indices):
    host = numpy.random.default_rng(49).integers(-10, 10, size).astype(dtype)
    a = cupy.asarray(host)
    sort = (compute_sort.cuda_compute_argsort if return_indices
            else compute_sort.cuda_compute_sort)
    ready = cupy.cuda.get_current_stream().record()
    with cupy.cuda.Stream(non_blocking=True) as stream:
        stream.wait_event(ready)
        result = sort(a, descending, algorithm=algorithm, overwrite_input=True)
        actual = cupy.asnumpy(result)
    if return_indices:
        order_values = host
        if descending:
            order_values = ~host if host.dtype.kind == 'b' else -host
        expected = numpy.argsort(order_values, kind='stable')
    else:
        assert result is a
        expected = _expected(host, descending)
    numpy.testing.assert_array_equal(actual, expected)


@pytest.mark.thread_unsafe(reason='Patches the workspace allocator.')
@pytest.mark.parametrize('dtype', ['int16', 'int32'])
@pytest.mark.parametrize('return_indices', [False, True])
def test_radix_misaligned_workspace(dtype, return_indices):
    host = numpy.random.default_rng(50).integers(-10, 10, 10007).astype(dtype)
    a = cupy.asarray(host)
    sort = (compute_sort.cuda_compute_argsort if return_indices
            else compute_sort.cuda_compute_sort)
    with mock.patch.object(cupy.cuda, 'alloc', _offset_allocator(1)):
        result = sort(a)
    expected = (numpy.argsort(host, kind='stable') if return_indices
                else numpy.sort(host, kind='stable'))
    numpy.testing.assert_array_equal(cupy.asnumpy(result), expected)
    numpy.testing.assert_array_equal(cupy.asnumpy(a), host)


def _native_float_radix_indices(host, descending, *, rotate_nans=False):
    bits = host.view(numpy.dtype(f'u{host.itemsize}'))
    sign = bits.dtype.type(1 << (8 * host.itemsize - 1))
    magnitude = bits & ~sign
    # Native CUB radix order includes NaN sign/payload bits, but treats both
    # signed zeros as equivalent.
    keys = numpy.where(bits & sign, ~bits, bits ^ sign)
    keys[magnitude == 0] = sign
    indices = numpy.argsort(~keys if descending else keys, kind='stable')
    if rotate_nans:
        infinity = numpy.array(numpy.inf, host.dtype).view(bits.dtype)
        is_nan = magnitude > infinity
        if not is_nan.all():
            leading = is_nan & ((bits & sign != 0) != descending)
            indices = numpy.roll(indices, -numpy.count_nonzero(leading))
    return indices


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('return_indices', [False, True])
def test_native_float_radix(dtype, descending, return_indices):
    dtype = numpy.dtype(dtype)
    unsigned = numpy.dtype(f'u{dtype.itemsize}')
    bits = numpy.random.default_rng(51).integers(
        0, numpy.iinfo(unsigned).max, 10007, dtype=unsigned, endpoint=True)
    sign = unsigned.type(1 << (8 * dtype.itemsize - 1))
    bits[:4] = [0, sign, sign - unsigned.type(1), sign + unsigned.type(1)]
    host = bits.view(dtype)
    indices = _native_float_radix_indices(host, descending)
    a = cupy.asarray(host)
    sort = (compute_sort.cuda_compute_argsort if return_indices
            else compute_sort.cuda_compute_sort)
    result = sort(a, descending, algorithm='radix_native')
    actual = cupy.asnumpy(result)
    if return_indices:
        numpy.testing.assert_array_equal(actual, indices)
    else:
        numpy.testing.assert_array_equal(actual.view(unsigned), bits[indices])
    numpy.testing.assert_array_equal(cupy.asnumpy(a).view(unsigned), bits)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'] + (
    ['bfloat16'] if cupy._util.BF16 is not None else []))
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('return_indices', [False, True])
@pytest.mark.parametrize('case,overwrite_input', [
    ('mixed', False), ('mixed_large', True), ('tail_only', True),
    ('all_nan', False), ('all_leading_nan', False), ('no_nan', True),
])
def test_native_float_radix_rotate(
        dtype, descending, return_indices, case, overwrite_input):
    dtype = numpy.dtype(dtype)
    unsigned = numpy.dtype(f'u{dtype.itemsize}')
    sign = 1 << (8 * dtype.itemsize - 1)
    infinity = int(numpy.array(numpy.inf, dtype).view(unsigned))
    quiet_bit = (infinity & -infinity) >> 1
    positive_nans = [infinity | 1, infinity | quiet_bit, infinity | 2]
    negative_nans = [sign | bits for bits in positive_nans]
    numbers = [sign, 0, 1, sign | 1, infinity, sign | infinity]
    if case == 'all_nan':
        pattern = negative_nans + positive_nans
    elif case == 'all_leading_nan':
        pattern = positive_nans if descending else negative_nans
    elif case == 'no_nan':
        pattern = numbers
    elif case == 'tail_only':
        pattern = numbers + (negative_nans if descending else positive_nans)
    else:
        pattern = numbers + negative_nans + positive_nans
    bits = numpy.array(pattern, dtype=unsigned)
    if case == 'mixed_large':
        bits = numpy.resize(bits, 10007)
        numpy.random.default_rng(52).shuffle(bits)

    indices = _native_float_radix_indices(
        bits.view(dtype), descending, rotate_nans=True)

    sort = (compute_sort.cuda_compute_argsort if return_indices
            else compute_sort.cuda_compute_sort)
    with cupy.cuda.Stream(non_blocking=True):
        a = cupy.asarray(bits).view(dtype)
        result = sort(a, descending, algorithm='radix_native_rotate',
                      overwrite_input=overwrite_input)
        if return_indices:
            numpy.testing.assert_array_equal(cupy.asnumpy(result), indices)
        else:
            assert (result is a) == overwrite_input
            numpy.testing.assert_array_equal(
                cupy.asnumpy(result.view(unsigned)), bits[indices])
        if not overwrite_input:
            numpy.testing.assert_array_equal(
                cupy.asnumpy(a.view(unsigned)), bits)


@pytest.mark.parametrize('itemsize', [2, 4, 8])
@pytest.mark.parametrize('misalign', ['none', 'source', 'destination'])
def test_native_float_radix_copy_zero_shift(itemsize, misalign):
    dtype = numpy.dtype(f'u{itemsize}')
    size = 10007  # Exercises the scalar tail for each vector width.
    source_start = int(misalign == 'source')
    destination_start = int(misalign == 'destination')
    with cupy.cuda.Stream(non_blocking=True):
        source = cupy.arange(1, size + 2, dtype=dtype)
        source = source[source_start:source_start + size]
        storage = cupy.zeros(size + 1, dtype=dtype)
        destination = storage[destination_start:destination_start + size]
        offset = cupy.zeros((), dtype=cupy.uint64)
        _, copy = compute_sort._radix_rotation_kernels(
            numpy.dtype('float32'), itemsize)
        copy((2,), (256,), (source, destination, numpy.uint64(size), offset,
                            True, False))
        expected = numpy.zeros(size + 1, dtype=dtype)
        expected[destination_start:destination_start + size] = numpy.arange(
            source_start + 1, source_start + size + 1,
            dtype=dtype)
        numpy.testing.assert_array_equal(storage.get(), expected)


def _check_float_1d(host, descending, return_indices, algorithm):
    sort = (compute_sort.cuda_compute_argsort if return_indices
            else compute_sort.cuda_compute_sort)
    bits_dtype = numpy.dtype(f'u{host.dtype.itemsize}')
    if algorithm == 'auto':
        indices = _native_float_radix_indices(
            host, descending, rotate_nans=True)
    else:
        with numpy.errstate(invalid='ignore'):
            indices = numpy.argsort(
                -host if descending else host, kind='stable')
    with cupy.cuda.Stream(non_blocking=True):
        a = cupy.asarray(host)
        result = sort(a, descending, algorithm=algorithm)
        actual = cupy.asnumpy(result)
        original = cupy.asnumpy(a)
    assert result is not a
    if return_indices:
        assert result.dtype == numpy.dtype(numpy.intp)
        numpy.testing.assert_array_equal(actual, indices)
    else:
        assert result.dtype == host.dtype
        numpy.testing.assert_array_equal(
            actual.view(bits_dtype), host[indices].view(bits_dtype))
    numpy.testing.assert_array_equal(
        original.view(bits_dtype), host.view(bits_dtype))


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('return_indices', [False, True])
@pytest.mark.parametrize('algorithm', ['auto', 'merge', 'radix_transform'])
def test_float_1d(dtype, descending, return_indices, algorithm):
    dtype = numpy.dtype(dtype)
    # Include signaling/quiet NaNs of both signs and different payloads,
    # infinities, signed zeros, subnormals, and repeated finite values.
    patterns = {
        2: [0xfc01, 0x7c01, 0xfe01, 0x7e02, 0x8000, 0,
            0xfc00, 0x7c00, 0x8001, 1, 0xfbff, 0x7bff, 0xbc00, 0x3c00],
        4: [0xff800001, 0x7f800001, 0xffc00001, 0x7fc00002, 0x80000000, 0,
            0xff800000, 0x7f800000, 0x80000001, 1,
            0xff7fffff, 0x7f7fffff, 0xbf800000, 0x3f800000],
        8: [0xfff0000000000001, 0x7ff0000000000001,
            0xfff8000000000001, 0x7ff8000000000002, 0x8000000000000000, 0,
            0xfff0000000000000, 0x7ff0000000000000, 0x8000000000000001, 1,
            0xffefffffffffffff, 0x7fefffffffffffff,
            0xbff0000000000000, 0x3ff0000000000000],
    }
    bits = numpy.resize(numpy.array(
        patterns[dtype.itemsize], dtype=f'u{dtype.itemsize}'), 10007)
    numpy.random.default_rng(47).shuffle(bits)
    _check_float_1d(bits.view(dtype), descending, return_indices, algorithm)


@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('return_indices', [False, True])
@pytest.mark.parametrize('algorithm', ['auto', 'merge', 'radix_transform'])
def test_float16_all_encodings(descending, return_indices, algorithm):
    bits = numpy.arange(1 << 16, dtype=numpy.uint16)
    numpy.random.default_rng(48).shuffle(bits)
    _check_float_1d(
        bits.view(numpy.float16), descending, return_indices, algorithm)


@pytest.mark.skipif(cupy._util.BF16 is None,
                    reason='bfloat16 is unavailable')
@pytest.mark.parametrize('algorithm,shape', [
    ('merge', (131072,)), ('merge', (2, 65536)), ('merge', (2, 2, 32768)),
    ('radix_transform', (131072,)),
    ('auto', (131072,)),
])
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('return_indices', [False, True])
def test_bfloat16_all_encodings(algorithm, shape, descending, return_indices):
    # Every encoding twice exercises NaN payloads/signs, signed zeros, and
    # stability across tiles.
    bits = numpy.tile(numpy.arange(65536, dtype=numpy.uint16), 2)
    numpy.random.default_rng(56).shuffle(bits)
    bits = bits.reshape(shape)
    host = bits.view(numpy.dtype('bfloat16'))
    if algorithm == 'auto':
        indices = _native_float_radix_indices(
            host, descending, rotate_nans=True)
    else:
        keys = host.astype(numpy.float32)
        indices = numpy.argsort(
            -keys if descending else keys, axis=-1, kind='stable')
    sort = (compute_sort.cuda_compute_argsort if return_indices
            else compute_sort.cuda_compute_sort)
    with cupy.cuda.Stream(non_blocking=True):
        a = cupy.asarray(host)
        result = sort(a, descending, algorithm=algorithm)
        assert result.dtype == (numpy.dtype(numpy.intp) if return_indices
                                else host.dtype)
        actual = cupy.asnumpy(result if return_indices
                              else result.view(cupy.uint16))
        numpy.testing.assert_array_equal(
            cupy.asnumpy(a.view(cupy.uint16)), bits)
    expected = (indices if return_indices
                else numpy.take_along_axis(bits, indices, axis=-1))
    numpy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('return_indices', [False, True])
def test_algorithm_selection(return_indices):
    sort = (compute_sort.cuda_compute_argsort if return_indices
            else compute_sort.cuda_compute_sort)
    for algorithm in ('invalid', 'radix'):
        with pytest.raises(ValueError, match='algorithm'):
            sort(cupy.ones(3), algorithm=algorithm)
    for algorithm in ('radix_native', 'radix_transform'):
        assert sort(cupy.ones(3, dtype='complex64'),
                    algorithm=algorithm) is None
        assert sort(cupy.ones((2, 3)), algorithm=algorithm) is None
    for dtype in ('bool', 'int32', 'float32') + (
            ('bfloat16',) if cupy._util.BF16 is not None else ()):
        for size in (0, 1):
            a = cupy.ones(size, dtype=dtype)
            for algorithm in ('merge', 'radix_native', 'radix_transform'):
                if dtype == 'bfloat16' and algorithm == 'radix_native':
                    assert sort(a, algorithm=algorithm) is None
                    continue
                result = sort(a, algorithm=algorithm)
                expected = (
                    numpy.zeros(size, dtype=numpy.intp) if return_indices
                    else numpy.ones(size, dtype=dtype))
                numpy.testing.assert_array_equal(
                    cupy.asnumpy(result), expected)


@pytest.mark.parametrize('dtype', sorted(compute_sort._SUPPORTED_DTYPES))
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('multi_row', [False, True])
def test_sort(dtype, descending, multi_row):
    rng = numpy.random.default_rng(42)
    shape = (2, 3, 7513) if multi_row else (7513,)
    host = rng.integers(0, 100, size=shape).astype(dtype)
    if host.dtype.kind == 'b':
        host = rng.integers(0, 2, size=shape).astype(dtype)
    elif host.dtype.kind == 'f' or host.dtype.name == 'bfloat16':
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
        result = compute_sort.cuda_compute_sort(a, descending)
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
    result = compute_sort.cuda_compute_sort(a, descending, algorithm='merge')
    numpy.testing.assert_array_equal(
        cupy.asnumpy(result).view(numpy.uint8),
        _expected(host, descending).view(numpy.uint8))


@pytest.mark.thread_unsafe(reason='Mocks the global merge-sort runner.')
@pytest.mark.parametrize('shape', [(0,), (3, 0), (0, 4), (1,), (2, 1)])
def test_empty_and_singleton(shape):
    a = cupy.empty(shape, dtype=numpy.float32)
    with mock.patch.object(compute_sort, '_run_merge_sort') as build:
        result = compute_sort.cuda_compute_sort(a)
        assert result is not a
        assert result.shape == a.shape
        assert result.dtype == a.dtype
        numpy.testing.assert_array_equal(cupy.asnumpy(result), cupy.asnumpy(a))
    build.assert_not_called()


def test_unsupported():
    assert compute_sort.cuda_compute_sort(numpy.ones(3)) is None
    assert compute_sort.cuda_compute_sort(cupy.ones(())) is None
    assert compute_sort.cuda_compute_sort(cupy.ones(10)[::2]) is None


@pytest.mark.thread_unsafe(reason='Mocks the global cuda.compute lookup.')
def test_unavailable():
    with mock.patch.object(
            compute_sort, '_get_cuda_compute', return_value=None):
        assert compute_sort.cuda_compute_sort(cupy.ones(3)) is None


@pytest.mark.parametrize('shape,dtype,algorithm', [
    ((3,), 'int32', 'auto'), ((1, 3), 'int32', 'auto'),
    ((3,), 'float32', 'auto'), ((3,), 'float32', 'radix_transform'),
])
@pytest.mark.parametrize('return_indices', [False, True])
@pytest.mark.parametrize('overwrite_input', [False, True])
def test_allocation_failure_leaves_input_unchanged(
        shape, dtype, algorithm, return_indices, overwrite_input):
    host = numpy.array([3, 1, 2], dtype=dtype).reshape(shape)
    a = cupy.asarray(host)
    # Warm compilation and caches before injecting execution-time failures.
    sort = (compute_sort.cuda_compute_argsort if return_indices
            else compute_sort.cuda_compute_sort)
    result = sort(a, algorithm=algorithm, overwrite_input=overwrite_input)
    assert result is not None
    cupy.cuda.get_current_stream().synchronize()
    a = cupy.asarray(host)
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
            result = sort(a, algorithm=algorithm,
                          overwrite_input=overwrite_input)
        if failure_injected or not overwrite_input:
            numpy.testing.assert_array_equal(cupy.asnumpy(a), host)
        if not failure_injected:
            assert failure_index > 0
            assert result is not None
            expected = (numpy.argsort(host, axis=-1, kind='stable')
                        if return_indices else numpy.sort(host, axis=-1))
            numpy.testing.assert_array_equal(cupy.asnumpy(result), expected)
            break
        assert result is None
        failure_index += 1


@pytest.mark.thread_unsafe(reason='Patches global cache and factories.')
@pytest.mark.parametrize('operation', ['sort', 'argsort', 'lexsort'])
@pytest.mark.parametrize('dtype', ['int32', 'float32'] + (
    ['bfloat16'] if cupy._util.BF16 is not None else []))
def test_cache_round_trip_and_shape_reuse(tmp_path, operation, dtype):
    compute = _cuda_compute_common._get_cuda_compute()
    backend = DiskKernelCacheBackend(str(tmp_path))
    errors = []

    def run():
        try:
            # Same ndim family and dtype: different row widths must reuse the
            # algorithm, since width is runtime state. 1-D has its own family.
            shapes = [(17,), (29,), (17,), (2, 17), (1, 34),
                      (3, 29), (2, 17)]
            if operation == 'lexsort':
                shapes = [(2, 17), (3, 17), (3, 29), (2, 17)]
            for shape in shapes:
                host = numpy.arange(numpy.prod(shape), dtype=dtype)
                host = host[::-1].copy().reshape(shape)
                # Reuse the algorithm across new allocations, streams, and
                # runtime operator state.
                with cupy.cuda.Stream(non_blocking=True):
                    a = cupy.asarray(host)
                    if operation == 'sort':
                        result = compute_sort.cuda_compute_sort(a)
                        expected = numpy.sort(host, axis=-1)
                    elif operation == 'argsort':
                        result = compute_sort.cuda_compute_argsort(a)
                        expected = numpy.argsort(
                            host, axis=-1, kind='stable')
                    else:
                        result = compute_sort.cuda_compute_lexsort(a)
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
        with (
            mock.patch.object(
                compute, 'make_merge_sort',
                wraps=compute.make_merge_sort) as build,
            mock.patch.object(
                compute, 'make_radix_sort',
                wraps=compute.make_radix_sort) as radix_build,
            mock.patch.object(
                compute, 'make_unary_transform',
                wraps=compute.make_unary_transform) as transform_build,
        ):
            in_new_thread()
            uses_radix = operation != 'lexsort'
            assert build.call_count == 1
            assert radix_build.call_count == int(uses_radix)
            transform_build.assert_not_called()
            if uses_radix:
                kwargs = radix_build.call_args.kwargs
                keys = kwargs['d_in_keys']
                assert isinstance(keys.current(), compute.ProxyArray)
                assert isinstance(keys.alternate(), compute.ProxyArray)
                values = kwargs['d_in_values']
                if operation == 'argsort':
                    assert isinstance(values.current(), compute.ProxyArray)
                    assert isinstance(values.alternate(), compute.ProxyArray)
                else:
                    assert values is None
        family = 'merge_sort' if operation == 'sort' else f'merge_{operation}'
        assert not list(tmp_path.glob('*.cc_radix_keys_transform'))
        if operation == 'lexsort':
            assert len(list(tmp_path.glob(f'*.cc_{family}'))) == 1
        else:
            family_1d = (f'radix_{operation}' if uses_radix else family)
            assert len(list(tmp_path.glob(f'*.cc_{family_1d}_1d'))) == 1
            assert len(list(tmp_path.glob(f'*.cc_{family}_nd'))) == 1
        with (
            mock.patch.object(
                compute, 'make_merge_sort',
                side_effect=AssertionError('Expected a disk-cache hit')),
            mock.patch.object(
                compute, 'make_radix_sort',
                side_effect=AssertionError('Expected a disk-cache hit')),
            mock.patch.object(
                compute, 'make_unary_transform',
                side_effect=AssertionError('Expected a disk-cache hit')),
        ):
            in_new_thread()


@pytest.mark.thread_unsafe(
    reason='Patches kernel cache and radix-sort factory.')
@pytest.mark.parametrize('return_indices', [False, True])
@pytest.mark.parametrize('dtype', ['bool', 'float32'] + (
    ['bfloat16'] if cupy._util.BF16 is not None else []))
def test_normalized_radix_cache_round_trip(tmp_path, return_indices, dtype):
    compute = _cuda_compute_common._get_cuda_compute()
    backend = DiskKernelCacheBackend(str(tmp_path))
    sort = (compute_sort.cuda_compute_argsort if return_indices
            else compute_sort.cuda_compute_sort)
    errors = []

    def run():
        try:
            # Shape and direction change only runtime state for encoded keys;
            # both directions sort those keys in ascending order.
            for size, descending in [(17, False), (29, True), (17, True)]:
                host = numpy.resize(numpy.array(
                    [3., numpy.nan, -0., 0., -1.], dtype=dtype), size)
                a = cupy.asarray(host)
                result = sort(a, descending, algorithm='radix_transform')
                order_values = host
                if descending:
                    order_values = ~host if dtype == 'bool' else -host
                expected = (numpy.argsort(
                    order_values, kind='stable')
                    if return_indices else _expected(host, descending))
                numpy.testing.assert_array_equal(
                    cupy.asnumpy(result), expected)
                numpy.testing.assert_array_equal(cupy.asnumpy(a), host)
        except BaseException as error:
            errors.append(error)

    with mock.patch('cupy.cuda.compiler._kernel_cache_backend', backend):
        for expected_builds in (1, 0):
            with (
                mock.patch.object(
                    compute, 'make_radix_sort',
                    wraps=compute.make_radix_sort) as build,
                mock.patch.object(
                    compute, 'make_unary_transform',
                    wraps=compute.make_unary_transform) as transform_build,
            ):
                thread = threading.Thread(target=run)
                thread.start()
                thread.join()
                if errors:
                    raise errors[0]
                assert build.call_count == expected_builds
                assert transform_build.call_count == 2 * expected_builds
    family = 'radix_argsort_1d' if return_indices else 'radix_sort_1d'
    assert len(list(tmp_path.glob(f'*.cc_{family}'))) == 1
    assert len(list(tmp_path.glob('*.cc_radix_keys_transform'))) == 2


@pytest.mark.parametrize('dtype', sorted(compute_sort._SUPPORTED_DTYPES))
def test_argsort(dtype):
    rng = numpy.random.default_rng(42)
    host = rng.integers(0, 10, size=(2, 7513)).astype(dtype)
    if host.dtype.kind == 'b':
        host = (host == 0)
    elif host.dtype.kind in 'fc' or host.dtype.name == 'bfloat16':
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
            result = compute_sort.cuda_compute_argsort(a, descending)
            assert result.dtype == numpy.dtype(numpy.intp)
            numpy.testing.assert_array_equal(
                cupy.asnumpy(result),
                numpy.argsort(order_values, axis=-1, kind='stable'))
        numpy.testing.assert_array_equal(cupy.asnumpy(a), data)


@pytest.mark.parametrize('dtype', sorted(compute_sort._SUPPORTED_DTYPES))
def test_lexsort(dtype):
    rng = numpy.random.default_rng(43)
    host = rng.integers(0, 5, size=(3, 7513)).astype(dtype)
    if host.dtype.kind == 'b':
        host = (host == 0)
    elif host.dtype.kind in 'fc' or host.dtype.name == 'bfloat16':
        host[:, :4] = [numpy.nan, -0., 0., numpy.nan]
        if host.dtype.kind == 'c':
            host[:, 4:8] = [complex(numpy.nan, 1), complex(1, numpy.nan),
                            complex(numpy.nan, numpy.nan), complex(1, -1)]
    a = cupy.asarray(host)
    result = compute_sort.cuda_compute_lexsort(a)
    assert result.dtype == numpy.dtype(numpy.intp)
    numpy.testing.assert_array_equal(cupy.asnumpy(result), numpy.lexsort(host))
    numpy.testing.assert_array_equal(cupy.asnumpy(a), host)


def test_indirect_sort_empty_and_singleton():
    for shape in [(0,), (2, 0), (0, 4), (1,), (2, 1)]:
        a = cupy.zeros(shape)
        result = compute_sort.cuda_compute_argsort(a)
        numpy.testing.assert_array_equal(
            cupy.asnumpy(result),
            numpy.argsort(cupy.asnumpy(a), kind='stable'))
    for shape in [(2, 0), (2, 1), (0, 5)]:
        a = cupy.zeros(shape)
        result = compute_sort.cuda_compute_lexsort(a)
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

    @pytest.mark.parametrize('size,use_radix', [
        (32767, False), (32768, False), (32769, True),
    ])
    @pytest.mark.parametrize('descending', [False, True])
    @pytest.mark.parametrize('return_indices', [False, True])
    def test_bool_auto_dispatch(self, size, use_radix, descending,
                                return_indices):
        host = numpy.random.default_rng(58).integers(0, 2, size).astype(bool)
        indices = numpy.argsort(
            ~host if descending else host, kind='stable')
        a = cupy.asarray(host)
        with (
            mock.patch.object(
                compute_sort, '_run_merge_sort',
                wraps=compute_sort._run_merge_sort) as merge,
            mock.patch.object(
                compute_sort, '_sort_radix_1d',
                wraps=compute_sort._sort_radix_1d) as radix,
            mock.patch.object(thrust, 'available', False),
        ):
            if return_indices:
                result = a.argsort(descending=descending)
                numpy.testing.assert_array_equal(cupy.asnumpy(a), host)
            else:
                assert a.sort(descending=descending) is None
                result = a
        assert merge.call_count == int(not use_radix)
        assert radix.call_count == int(use_radix)
        if use_radix:
            assert radix.call_args.kwargs['native'] is False
        numpy.testing.assert_array_equal(
            cupy.asnumpy(result), indices if return_indices else host[indices])

    @pytest.mark.skipif(cupy._util.BF16 is None,
                        reason='bfloat16 is unavailable')
    @pytest.mark.parametrize('operation', ['sort', 'argsort', 'lexsort'])
    def test_bfloat16_dispatch(self, operation):
        host = numpy.array([
            [3., numpy.nan, -0., 0., -1.],
            [numpy.nan, 2., 0., -0., 1.],
        ], dtype='bfloat16')
        a = cupy.asarray(host)
        # Disallow fallback so success demonstrates accelerator dispatch.
        with mock.patch.object(thrust, 'available', False):
            result = getattr(cupy, operation)(a)
        if operation == 'sort':
            assert result.dtype == host.dtype
            actual = cupy.asnumpy(result.view(cupy.uint16))
            expected = _expected(host, False).view(numpy.uint16)
        else:
            reference = host.astype(numpy.float32)
            actual = cupy.asnumpy(result)
            expected = (numpy.lexsort(reference) if operation == 'lexsort'
                        else numpy.argsort(reference, kind='stable'))
        numpy.testing.assert_array_equal(actual, expected)
        numpy.testing.assert_array_equal(
            cupy.asnumpy(a.view(cupy.uint16)), host.view(numpy.uint16))

    @pytest.mark.parametrize('dtype', [
        'int16', 'int32', 'float16', 'float32', 'float64',
    ] + (['bfloat16'] if cupy._util.BF16 is not None else []))
    @pytest.mark.parametrize('size', [2048, 10007])
    @pytest.mark.parametrize('return_indices', [False, True])
    def test_radix_1d_dispatch(self, dtype, size, return_indices):
        host = numpy.random.default_rng(52).integers(
            -10, 10, size).astype(dtype)
        a = cupy.asarray(host)
        view = a.view()
        with (
            mock.patch.object(thrust, 'available', False),
            mock.patch.object(
                compute_sort, '_sort_radix_1d',
                wraps=compute_sort._sort_radix_1d) as radix,
        ):
            if return_indices:
                result = a.argsort()
                expected = numpy.argsort(host, kind='stable')
                numpy.testing.assert_array_equal(cupy.asnumpy(view), host)
            else:
                assert a.sort() is None
                result = a
                expected = numpy.sort(host, kind='stable')
                numpy.testing.assert_array_equal(cupy.asnumpy(view), expected)
        radix.assert_called_once()
        is_float = host.dtype.kind == 'f' or dtype == 'bfloat16'
        assert radix.call_args.kwargs['rotate_nans'] is is_float
        numpy.testing.assert_array_equal(cupy.asnumpy(result), expected)

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
                compute_sort, 'cuda_compute_sort',
                wraps=compute_sort.cuda_compute_sort) as accelerated:
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

    @pytest.mark.parametrize('dtype', ['int32', 'float32', 'complex128'])
    @pytest.mark.parametrize('use_method', [False, True])
    def test_argsort_dispatch(self, dtype, use_method):
        rng = numpy.random.default_rng(44)
        host = rng.integers(0, 5, (3, 37)).astype(dtype)
        if host.dtype.kind in 'fc':
            host[0, :3] = [numpy.nan, 0., -0.]
        a = cupy.asarray(host).T
        host = host.T
        for axis in [-1, 0, None]:
            for descending in [False, True]:
                with mock.patch.object(
                        compute_sort, 'cuda_compute_argsort',
                        wraps=compute_sort.cuda_compute_argsort,
                ) as accelerated:
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
                compute_sort, 'cuda_compute_lexsort',
                wraps=compute_sort.cuda_compute_lexsort) as accelerated:
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
        entry = ('cuda_compute_sort' if operation == 'sort'
                 else f'cuda_compute_{operation}')
        patches = {
            'disabled': mock.patch.object(compute_sort, builder),
            'declined': mock.patch.object(
                compute_sort, entry, return_value=None),
            'missing': mock.patch.object(
                compute_sort, '_get_cuda_compute', return_value=None),
            'oom': mock.patch.object(
                compute_sort, builder,
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
