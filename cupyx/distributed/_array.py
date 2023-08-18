import enum
from typing import Any, Optional

import cupy
import numpy

from numpy.typing import ArrayLike
from cupy.typing import NDArray


def _extgcd(a: int, b: int) -> tuple[int, int]:
    """Return (g, x) with g = gcd(a, b), ax + by = g - ax.
    a, b > 0 is assumed."""
    # c - ax - by = 0  ...  (1)
    # d - au - bv = 0  ...  (2)
    c, d = a, b
    x, u = 1, 0
    # y, v = 0, 1

    # Apply Euclid's algorithm to (c, d)
    while d:
        r = c // d
        # (1), (2) = (2), (1) - (2) * r
        c, d = d, c - d * r
        x, u = u, x - u * r
        # y, v = v, y - u * r

    return c, x


def _slice_intersection(a: slice, b: slice, length: int) -> Optional[slice]:
    """Return the intersection of slice a and b. None if they are disjoint."""
    a_start, a_stop, a_step = a.indices(length)
    b_start, b_stop, b_step = b.indices(length)

    # a_step * x + b_step * y == g  ...  (1)
    g, x = _extgcd(a_step, b_step)
    if (b_start - a_start) % g != 0:
        return None

    # c is the intersection of a, b
    # c_step == lcm(a_step, b_step)
    c_step = a_step // g * b_step

    # Multiply (1) by (b_start - a_start) // g
    # ==> a_step * a_skip - b_step * b_skip == b_start - a_start
    #     a_start + a_step * a_skip == b_start + b_step * b_skip
    a_skip = x * ((b_start - a_start) // g) % (c_step // a_step)
    c_start = a_start + a_step * a_skip
    if c_start < b_start:
        c_start += ((b_start - c_start - 1) // c_step + 1) * c_step

    c_stop = min(a_stop, b_stop)
    if c_start < c_stop:
        return slice(c_start, c_stop, c_step)
    else:
        return None


def _index_for_subslice(a: slice, sub: slice, length: int) -> slice:
    """Return slice c such that array[a][c] == array[sub].
    sub should be contained in a."""
    a_start, a_stop, a_step = a.indices(length)
    sub_start, sub_stop, sub_step = sub.indices(length)

    c_start = (sub_start - a_start) // a_step
    # a_start + a_step * (c_stop - 1) < sub_stop
    c_stop = (sub_stop - a_start - 1) // a_step + 1
    c_step = sub_step // a_step

    return slice(c_start, c_stop, c_step)


def _index_intersection(
        a_idx: tuple[slice, ...], b_idx: tuple[slice, ...],
        shape: tuple[int, ...]) -> Optional[tuple[slice, ...]]:
    """Return None if empty."""
    ndim = len(shape)
    assert len(a_idx) == len(b_idx) == ndim
    result = tuple(_slice_intersection(a, b, length)
                   for a, b, length in zip(a_idx, b_idx, shape))
    if None in result:
        return None
    else:
        return result


def _index_for_subindex(
        a_idx: tuple[slice, ...], sub_idx: tuple[slice, ...],
        shape: tuple[int, ...]) -> tuple[slice, ...]:
    ndim = len(shape)
    assert len(a_idx) == len(sub_idx) == ndim

    return tuple(_index_for_subslice(a, sub, length)
                   for a, sub, length in zip(a_idx, sub_idx, shape))


# Temporary helper function.
# Should be removed after implementing indexing
def _shape_after_indexing(
        outer_shape: tuple[int, ...],
        idx: tuple[slice, ...]) -> tuple[int, ...]:
    shape = list(outer_shape)
    for i in range(len(idx)):
        start, stop, step = idx[i].indices(shape[i])
        shape[i] = (stop - start - 1) // step + 1
    return tuple(shape)


def _convert_chunk_idx_to_slice(
        shape: tuple[int, ...], idx: Any) -> tuple[slice, ...]:
    """Convert idx to type tuple[slice, ...] with all nonnegative indices and
    length == ndim. Raise if empty or invalid.

    Negative slice steps are not allowed, because this function is for
    representing chunks, e.g. the indices in device_mapping."""

    if not isinstance(idx, tuple):
        idx = idx,

    ndim = len(shape)
    if len(idx) > ndim:
        raise IndexError(
            'too many indices for array:'
            f' array is {ndim}-dimensional, but {len(idx)} were indexed')
    idx = idx + (slice(None),) * (ndim - len(idx))

    new_idx = []
    for i in range(ndim):
        if isinstance(idx[i], int):
            if idx[i] >= shape[i]:
                raise IndexError(
                    f'Index {idx[i]} is out of bounds'
                    f' for axis {i} with size {shape[i]}')
            new_idx.append(slice(idx[i], idx[i] + 1))
        elif isinstance(idx[i], slice):
            start, stop, step = idx[i].indices(shape[i])
            if step == 0:
                raise ValueError('Slice step must be nonzero')
            if step < 0:
                raise ValueError(
                    'The indices for a chunk cannot have negative slice steps.')
            if start == stop:
                raise ValueError(f'The index is empty on axis {i}')
            new_idx.append(slice(start, stop, step))
        else:
            raise ValueError(f'Invalid index on axis {i}')

    return tuple(new_idx)


class _MultiDeviceDummyMemory(cupy.cuda.Memory):
    pass


class _MultiDeviceDummyPointer(cupy.cuda.MemoryPointer):
    @property
    def device(self):
        # This override is needed to assign an invalid device id
        # Since the array is not residing in a single device now
        return cupy.cuda.device.Device(-1)


class _DistributedArray(cupy.ndarray):
    _chunks: dict[int, NDArray]
    # Values of _device_mapping must have lenth == ndim
    _device_mapping: dict[int, tuple[slice, ...]]

    def __new__(cls, shape, dtype, chunks, device_mapping):
        mem = _MultiDeviceDummyMemory(0)
        memptr = _MultiDeviceDummyPointer(mem, 0)
        obj = super().__new__(cls, shape, dtype, memptr=memptr)
        obj._chunks = chunks
        obj._device_mapping = device_mapping
        obj._mem = mem
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._chunks = getattr(obj, 'chunks', None)
        self._mem = getattr(obj, 'mem', None)

    def _get_execution_devices(self, dist_args):
        devices = set()
        for _, arg in dist_args:
            # The key of chunks is the device id
            for dev in arg._chunks:
                devices.add(dev)
        return devices

    def _get_chunk(self, i):
        return self._chunks[i]

    def _prepare_args(self, dist_args, regular_args, device):
        # Dist arrays must have chunks of compatible shapes, otherwise
        # hard error.
        # In case that they are of different, but broadcastable shapes
        # Data movement may be needed
        # Currently: Support only same shape chunks
        args = []
        c_shape = None
        for (i, arg) in dist_args:
            chunk = arg._get_chunk(device)
            args.append((i, chunk))
            if c_shape is None:
                c_shape = chunk.shape
            # TODO(ecastill) check if broadcastable, the array must have been
            # split in the same axis?
            if chunk.shape != c_shape:
                raise RuntimeError(
                    'Operating distributed arrays of different chunk sizes'
                    ' together is not supported')

        # Case of X.T and other data movement requiring cases not supported
        # TODO(ecastill) add support for operands being non distributed arrays
        # 1. Check if the regular arrays are on the specified device or
        #    peer access is enabled
        # 2. Check that their shape is compatible with the chunks
        #    distributed arrays
        # 3. Create views of this array and copy to the given device if needed
        #    so that the chunks in the distributed operate with the right slice
        if len(regular_args) > 0:
            raise RuntimeError(
                'Mix `cupy.ndarray` with distributed arrays is not currently'
                'supported')

        return args

    def _execute_kernel(self, kernel, args, kwargs):
        distributed_arrays = []
        regular_arrays = []
        for i, arg in enumerate(args):
            if isinstance(arg, _DistributedArray):
                distributed_arrays.append((i, arg))
            elif isinstance(arg, cupy.ndarray):
                regular_arrays.append((i, arg))

        # Do it for kwargs too
        for k, arg in kwargs.items():
            if isinstance(arg, _DistributedArray):
                distributed_arrays.append((k, arg))
            elif isinstance(arg, cupy.ndarray):
                regular_arrays.append((k, arg))

        args = list(args)
        devices = self._get_execution_devices(distributed_arrays)
        dev_outs = {}
        dtype = None
        for dev in devices:
            array_args = self._prepare_args(
                distributed_arrays, regular_arrays, dev)
            for (i, arg) in array_args:
                if isinstance(i, int):
                    args[i] = arg
                else:
                    kwargs[i] = arg
            with cupy.cuda.Device(dev):
                out = kernel(*args, **kwargs)
                dtype = out.dtype
                dev_outs[dev] = out

        for out in dev_outs.values():
            if not isinstance(out, cupy.ndarray):
                raise RuntimeError(
                    'kernels returning other than single array not supported')

        # Elementwise operations preserve device_mapping
        return _DistributedArray(
            self.shape, dtype, dev_outs, self._device_mapping)

    def __cupy_override_elementwise_kernel__(self, kernel, *args, **kwargs):
        # This defines a protocol to be called from elementwise kernel
        # to override some of the ops done there
        outs = self._execute_kernel(kernel, args, kwargs)
        return outs

    def _write_view_to_view(
            self, src_array: NDArray,
            dst_array: NDArray, dst_idx: tuple[slice, ...]) -> None:
        with cupy.cuda.Device(dst_array.device.id):
            if src_array.device.id == dst_array.device.id:
                dst_array[dst_idx] = src_array
            else:
                # TODO: Try GPUDirect p2p access, NCCL send/recv
                dst_array[dst_idx] = cupy.array(cupy.asnumpy(src_array))

    def _apply_and_update_chunks(
            self, func, shape,
            a_dev, a_chunk, a_idx, b_dev, b_chunk, b_idx):
        intersection = _index_intersection(a_idx, b_idx, shape)
        if intersection is None:
            return
        a_new_idx = _index_for_subindex(a_idx, intersection, shape)
        b_new_idx = _index_for_subindex(b_idx, intersection, shape)
        with cupy.cuda.Device(a_dev):
            b_chunk_copied = cupy.empty_like(b_chunk[b_new_idx])
            self._write_view_to_view(
                b_chunk[b_new_idx], b_chunk_copied, (slice(None),))
            result = func(a_chunk[a_new_idx], b_chunk_copied)
            a_chunk[a_new_idx] = result
        self._write_view_to_view(result, b_chunk, b_new_idx)

    def _send_intersection(
            self,
            src_array: NDArray, src_idx: tuple[slice, ...],
            dst_array: NDArray, dst_idx: tuple[slice, ...]) -> None:
        """Write the intersection of chunks src_idx and dst_idx to the
        corresponding entries of dst_array. Note dst_array == self[dst_idx]."""

        # intersection == src_array[src_new_idx] == dst_array[dst_new_idx]
        intersection = _index_intersection(src_idx, dst_idx, self.shape)
        if intersection is None:
            return
        src_new_idx = _index_for_subindex(src_idx, intersection, self.shape)
        dst_new_idx = _index_for_subindex(dst_idx, intersection, self.shape)
        self._write_view_to_view(src_array[src_new_idx], dst_array, dst_new_idx)

    def __cupy_override_reduction_kernel__(
            self, kernel, axis, dtype, out, keepdims):
        # Assumption: kernel(x, y) == kernel(y, x) && kernel(x, x) == x e.g. max
        if out is not None:
            raise RuntimeError('Argument `out` is not supported')
        if keepdims:
            raise RuntimeError('`keepdims` is not supported')
        if kernel.name != 'cupy_max':
            raise RuntimeError(f'Unsupported kernel: {kernel.name}')

        shape = self.shape[:axis] + self.shape[axis+1:]
        chunks = {}
        device_mapping = {}
        # TODO: Parallelize this loop
        for dev, chunk in self._chunks.items():
            idx = self._device_mapping[dev]
            with cupy.cuda.Device(dev):
                chunks[dev] = kernel(chunk, axis=axis, dtype=dtype)
            device_mapping[dev] = idx[:axis] + idx[axis+1:]

        if kernel.name == 'cupy_max':
            func = cupy.maximum

        # Apply chunk src to other chunks with greater device ids
        # TODO: Parallelize
        for src_dev, src_chunk in chunks.items():
            src_idx = device_mapping[src_dev]
            for dst_dev, dst_chunk in chunks.items():
                if src_dev >= dst_dev:
                    continue
                dst_idx = device_mapping[dst_dev]
                self._apply_and_update_chunks(
                    func, shape,
                    src_dev, src_chunk, src_idx,
                    dst_dev, dst_chunk, dst_idx)

        return _DistributedArray(shape, dtype, chunks, device_mapping)

    def reshard(
        self, device_mapping: dict[int, Any]
    ) -> '_DistributedArray':
        old_mapping = self._device_mapping

        new_mapping = {dev: _convert_chunk_idx_to_slice(self.shape, idx)
                       for dev, idx in device_mapping.items()}

        new_chunks = {}
        for dst_dev, dst_idx in new_mapping.items():
            with cupy.cuda.Device(dst_dev):
                # dst_array = cupy.empty_like(self[dst_idx])
                dst_shape = _shape_after_indexing(self.shape, dst_idx)
                dst_array = cupy.empty(dst_shape)
            for src_dev, src_idx in old_mapping.items():
                # TODO: In Replica mode, ideally we want to avoid data
                # forwarding on elements that have already been forwarded from
                # another chunk. This must take into consideration various ways
                # chunks can overlap.
                src_array = self._chunks[src_dev]
                self._send_intersection(src_array, src_idx, dst_array, dst_idx)
            new_chunks[dst_dev] = dst_array

        return _DistributedArray(
            self.shape, self.dtype, new_chunks, new_mapping)

    def asnumpy(self):
        np_array = numpy.empty(self.shape)
        for dev, chunk_idx in self._device_mapping.items():
            # Multiple writes to a single element can happen
            # This is fine since we only support replica mode now
            np_array[chunk_idx] = cupy.asnumpy(self._chunks[dev])
        return np_array


class Mode(enum.Enum):
    replica = 1


def distributed_array(
        array: ArrayLike,
        device_mapping: dict[int, Any],
        mode: Mode = Mode.replica):
    if not isinstance(array, (numpy.ndarray, cupy.ndarray)):
        array = numpy.array(array)

    device_mapping = {dev: _convert_chunk_idx_to_slice(array.shape, idx)
                      for dev, idx in device_mapping.items()}

    cp_chunks = {}
    for dev, chunk_idx in device_mapping.items():
        if isinstance(array, cupy.ndarray):
            chunk = cupy.ascontiguousarray(array[chunk_idx])
        else:
            chunk = array[chunk_idx]
        with cupy.cuda.Device(dev):
            cp_chunks[dev] = cupy.array(chunk)
    return _DistributedArray(
        array.shape, array.dtype, cp_chunks, device_mapping)

