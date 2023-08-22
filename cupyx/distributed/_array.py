from dataclasses import dataclass
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


def _convert_chunk_idx_to_slices(
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


@dataclass
class OpMode:
    """None as Mode represents replica mode.
    func(x, x) == x must hold if identity is None."""
    func: cupy.ufunc
    numpy_func: numpy.ufunc
    identity: Any


Mode = Optional[OpMode]


_sum_mode = OpMode(cupy.add, numpy.add, 0)
_replica_mode = None


class _DistributedArray(cupy.ndarray):
    _chunks: dict[int, NDArray]
    # Values of _device_mapping must have lenth == ndim
    _device_mapping: dict[int, tuple[slice, ...]]
    _mode: Mode

    def __new__(cls, shape, dtype, chunks, device_mapping, mode):
        mem = _MultiDeviceDummyMemory(0)
        memptr = _MultiDeviceDummyPointer(mem, 0)
        obj = super().__new__(cls, shape, dtype, memptr=memptr)
        obj._chunks = chunks
        obj._device_mapping = device_mapping
        obj._mode = mode
        obj._mem = mem
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._chunks = getattr(obj, 'chunks', None)
        self._mem = getattr(obj, 'mem', None)

    @property
    def mode(self):
        return self._mode

    @property
    def device_mapping(self):
        return self._device_mapping

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
                distributed_arrays.append((i, arg.to_replica_mode()))
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
            self.shape, dtype, dev_outs, self._device_mapping, _replica_mode)

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
                dst_array[dst_idx] = src_array.copy()

    def _apply_and_update_chunks(
            self, op_mode, shape,
            src_chunk, src_idx, dst_chunk, dst_idx):
        intersection = _index_intersection(src_idx, dst_idx, shape)
        if intersection is None:
            return
        src_new_idx = _index_for_subindex(src_idx, intersection, shape)
        dst_new_idx = _index_for_subindex(dst_idx, intersection, shape)

        with cupy.cuda.Device(dst_chunk.device.id):
            src_chunk_copied = cupy.empty_like(src_chunk[src_new_idx])
            self._write_view_to_view(
                src_chunk[src_new_idx], src_chunk_copied, (slice(None),))

            dst_chunk[dst_new_idx] = op_mode.func(
                src_chunk_copied, dst_chunk[dst_new_idx])

        if op_mode.identity is not None:
            with cupy.cuda.Device(src_chunk.device.id):
                src_chunk[src_new_idx] = op_mode.identity

    def _all_reduce_intersections(
            self, op_mode, shape, chunks, device_mapping):
        chunks_list = list(chunks.items())

        # TODO: Parallelize
        for i in range(len(chunks_list)):
            src_dev, src_chunk = chunks_list[i]
            src_idx = device_mapping[src_dev]

            for j in range(i + 1, len(chunks_list)):
                dst_dev, dst_chunk = chunks_list[j]
                dst_idx = device_mapping[dst_dev]

                self._apply_and_update_chunks(
                    op_mode, shape,
                    src_chunk, src_idx, dst_chunk, dst_idx)

        # TODO: Parallelize
        for j in range(len(chunks_list)):
            src_dev, src_chunk = chunks_list[j]
            src_idx = device_mapping[src_dev]

            for i in range(j):
                dst_dev, dst_chunk = chunks_list[i]
                dst_idx = device_mapping[dst_dev]

                self._send_intersection(
                    shape, src_chunk, src_idx, dst_chunk, dst_idx)

    def _send_intersection(
            self, shape: tuple[int, ...],
            src_chunk: NDArray, src_idx: tuple[slice, ...],
            dst_chunk: NDArray, dst_idx: tuple[slice, ...]) -> None:
        """Write the intersection of chunks src_idx and dst_idx to the
        corresponding entries of dst_chunk. Note dst_chunk == self[dst_idx]."""

        # intersection == src_chunk[src_new_idx] == dst_chunk[dst_new_idx]
        intersection = _index_intersection(src_idx, dst_idx, shape)
        if intersection is None:
            return
        src_new_idx = _index_for_subindex(src_idx, intersection, shape)
        dst_new_idx = _index_for_subindex(dst_idx, intersection, shape)
        self._write_view_to_view(src_chunk[src_new_idx], dst_chunk, dst_new_idx)

    def _set_identity_on_intersection(
            self, shape: tuple[int, ...], identity,
            a_chunk: NDArray, a_idx: tuple[slice, ...],
            b_idx: tuple[slice, ...]) -> None:
        intersection = _index_intersection(a_idx, b_idx, shape)
        if intersection is None:
            return
        a_new_idx = _index_for_subindex(a_idx, intersection, shape)
        with cupy.cuda.Device(a_chunk.device.id):
            a_chunk[a_new_idx] = identity

    def __cupy_override_reduction_kernel__(
            self, kernel, axis, dtype, out, keepdims):
        # Assumption: kernel(x, y) == kernel(y, x) && kernel(x, x) == x e.g. max
        if out is not None:
            raise RuntimeError('Argument `out` is not supported')
        if keepdims:
            raise RuntimeError('`keepdims` is not supported')

        if kernel.name == 'cupy_max':
            func = cupy.maximum
            numpy_func = numpy.maximum
            identity = None
            chunks = self._replica_mode_chunks()
        elif kernel.name == 'cupy_min':
            func = cupy.minimum
            numpy_func = numpy.minimum
            identity = None
            chunks = self._replica_mode_chunks()
        elif kernel.name == 'cupy_sum':
            func = cupy.add
            numpy_func = numpy.add
            identity = 0
            chunks = self._op_mode_chunks(identity)
        elif kernel.name == 'cupy_prod':
            func = cupy.multiply
            numpy_func = numpy.multiply
            identity = 1
            chunks = self._op_mode_chunks(identity)
        else:
            raise RuntimeError(f'Unsupported kernel: {kernel.name}')

        shape = self.shape[:axis] + self.shape[axis+1:]
        new_chunks = {}
        device_mapping = {}
        # TODO: Parallelize this loop
        for dev, chunk in chunks.items():
            idx = self._device_mapping[dev]
            with cupy.cuda.Device(dev):
                new_chunks[dev] = kernel(chunk, axis=axis, dtype=dtype)
            device_mapping[dev] = idx[:axis] + idx[axis+1:]

        mode = OpMode(func, numpy_func, identity)
        return _DistributedArray(
            shape, dtype, new_chunks, device_mapping, mode)

    def _copy_chunks(self):
        chunks = {}
        for dev, chunk in self._chunks.items():
            with cupy.cuda.Device(dev):
                chunks[dev] = chunk.copy()
        return chunks

    def _replica_mode_chunks(self) -> dict[int, NDArray]:
        """Does not recessarily copy."""
        if self._mode is _replica_mode:
            return self._chunks

        chunks = self._copy_chunks()
        self._all_reduce_intersections(
            self._mode, self.shape, chunks, self._device_mapping)
        return chunks


    def _op_mode_chunks(self, identity) -> dict[int, NDArray]:
        chunks = self._replica_mode_chunks()

        chunks_list = list(chunks.items())
        new_chunks = {}
        # TODO: Parallelize
        for i in range(len(chunks_list)):
            a_dev, a_chunk = chunks_list[i]
            with cupy.cuda.Device(a_dev):
                new_a_chunk = a_chunk.copy()
            a_idx = self._device_mapping[a_dev]

            for j in range(i + 1, len(chunks_list)):
                b_dev, _ = chunks_list[j]
                b_idx = self._device_mapping[b_dev]

                self._set_identity_on_intersection(
                    self.shape, identity, new_a_chunk, a_idx, b_idx)

            new_chunks[a_dev] = new_a_chunk

        return new_chunks

    def to_replica_mode(self) -> '_DistributedArray':
        """Does not recessarily copy."""
        chunks = self._replica_mode_chunks()
        return _DistributedArray(
            self.shape, self.dtype, chunks, self._device_mapping, _replica_mode)

    def change_mode(self, mode) -> '_DistributedArray':
        if self._mode is mode:
            return self

        chunks = self._copy_chunks()
        self._all_reduce_intersections(
            self._mode, self.shape, chunks, self._device_mapping)
        return _DistributedArray(
            self.shape, self.dtype, chunks, self._device_mapping, _replica_mode)

    def reshard(
        self, device_mapping: dict[int, Any]
    ) -> '_DistributedArray':
        # TODO: implement another version that reshards within the SUM mode
        arr = self.to_replica_mode()

        old_mapping = arr._device_mapping

        new_mapping = {dev: _convert_chunk_idx_to_slices(arr.shape, idx)
                       for dev, idx in device_mapping.items()}

        new_chunks = {}
        for dst_dev, dst_idx in new_mapping.items():
            with cupy.cuda.Device(dst_dev):
                # dst_array = cupy.empty_like(arr[dst_idx])
                dst_shape = _shape_after_indexing(arr.shape, dst_idx)
                dst_array = cupy.empty(dst_shape)
            for src_dev, src_idx in old_mapping.items():
                # TODO: In Replica mode, ideally we want to avoid data
                # forwarding on elements that have already been forwarded from
                # another chunk. This must take into consideration various ways
                # chunks can overlap.
                src_array = arr._chunks[src_dev]
                arr._send_intersection(
                    arr.shape, src_array, src_idx, dst_array, dst_idx)
            new_chunks[dst_dev] = dst_array

        return _DistributedArray(
            arr.shape, arr.dtype, new_chunks, new_mapping, _replica_mode)

    def asnumpy(self):
        if self._mode is _replica_mode or self._mode.identity is None:
            np_array = numpy.empty(self.shape)
        else:
            np_array = numpy.full(
                self.shape, self.mode.identity, dtype=self.dtype)

        for dev, chunk_idx in self._device_mapping.items():
            if self._mode is _replica_mode:
                np_array[chunk_idx] = cupy.asnumpy(self._chunks[dev])
            else:
                self._mode.numpy_func(
                    np_array[chunk_idx], cupy.asnumpy(self._chunks[dev]),
                    out=np_array[chunk_idx])

        return np_array


def distributed_array(
        array: ArrayLike,
        device_mapping: dict[int, Any],
        mode: Mode = _replica_mode):
    if not isinstance(array, (numpy.ndarray, cupy.ndarray)):
        array = numpy.array(array)
    elif mode is not _replica_mode:
        array = array.copy()

    device_mapping = {dev: _convert_chunk_idx_to_slices(array.shape, idx)
                      for dev, idx in device_mapping.items()}

    cp_chunks = {}
    for dev, chunk_idx in device_mapping.items():
        if isinstance(array, cupy.ndarray):
            chunk = cupy.ascontiguousarray(array[chunk_idx])
        else:
            chunk = array[chunk_idx]
        with cupy.cuda.Device(dev):
            cp_chunks[dev] = cupy.array(chunk)
        if mode is not _replica_mode and mode.identity is not None:
            array[chunk_idx] = mode.identity
    return _DistributedArray(
        array.shape, array.dtype, cp_chunks, device_mapping, mode)

