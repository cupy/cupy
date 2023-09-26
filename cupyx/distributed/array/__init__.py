from typing import Any, Callable, Final, Iterable, Optional, TypeGuard, Union
import functools
from itertools import chain

from cupyx.distributed.array import _chunk
from cupyx.distributed.array._chunk import _Chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array._data_transfer import NcclCommunicator
from cupyx.distributed.array import _elementwise
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _reduction
from cupyx.distributed.array import linalg

import numpy
from numpy.typing import ArrayLike, DTypeLike

import cupy
from cupy.cuda import Device, Stream, get_current_stream


@functools.lru_cache
def _min_value_of(dtype):
    if dtype.kind in 'biu':
        return dtype.type(cupy.iinfo(dtype).min)
    elif dtype.kind in 'f':
        return dtype.type(-cupy.inf)
    else:
        raise RuntimeError(f'Unsupported type: {dtype}')


@functools.lru_cache
def _max_value_of(dtype):
    if dtype.kind in 'biu':
        return dtype.type(cupy.iinfo(dtype).max)
    elif dtype.kind in 'f':
        return dtype.type(cupy.inf)
    else:
        raise RuntimeError(f'Unsupported type: {dtype}')


@functools.lru_cache
def _zero_of(dtype):
    return dtype.type(0)


@functools.lru_cache
def _one_of(dtype):
    return dtype.type(1)


class _OpMode:
    func: cupy.ufunc
    numpy_func: numpy.ufunc
    idempotent: bool
    identity_of: Callable

    def __init__(
        self, func_name: str, idempotent: bool, identity_of: Callable,
    ) -> None:
        try:
            self.func = getattr(cupy, func_name)
            self.numpy_func = getattr(numpy, func_name)
        except AttributeError:
            raise RuntimeError('No such function exists')

        self.idempotent = idempotent
        self.identity_of = identity_of


_Mode = Optional[_OpMode]


_REPLICA_MODE: Final[_Mode] = None


def _is_op_mode(mode: _Mode) -> TypeGuard[_OpMode]:
    return mode is not _REPLICA_MODE


_MODES: Final[dict[str, _Mode]] = {
    'replica': _REPLICA_MODE,
    'min':  _OpMode('minimum',  True,  _max_value_of),
    'max':  _OpMode('maximum',  True,  _min_value_of),
    'sum':  _OpMode('add',      False, _zero_of),
    'prod': _OpMode('multiply', False, _one_of),
}


class _MultiDeviceDummyMemory(cupy.cuda.Memory):
    pass


class _MultiDeviceDummyPointer(cupy.cuda.MemoryPointer):
    @property
    def device(self) -> Device:
        # This override is needed to assign an invalid device id
        # Since the array is not residing in a single device now
        return Device(-1)


def all_chunks(
    chunks_map: dict[int, list[_Chunk]],
) -> Iterable[tuple[int, _Chunk]]:
    for dev, chunks in chunks_map.items():
        for chunk in chunks:
            yield dev, chunk


class DistributedArray(cupy.ndarray):
    """Multi-dimensional array distributed across multiple CUDA devices.

    This class implements some elementary operations that :class:`cupy.ndarray`
    provides. The array content is split into chunks, contiguous arrays
    corresponding to slices of the original array. Note that one device can hold
    multiple chunks.
    """

    _chunks_map: dict[int, list[_Chunk]]
    _streams: dict[int, Stream]
    _mode: _Mode
    _comms: dict[int, NcclCommunicator]

    def __new__(
        cls, shape: tuple[int, ...], dtype: DTypeLike,
        chunks_map: dict[int, list[_Chunk]],
        mode: Union[str, _Mode] = _REPLICA_MODE,
        comms: Optional[dict[int, NcclCommunicator]] = None,
    ) -> 'DistributedArray':
        """Instantiate a distributed array using arguments for its attributes.

        :func:`~cupy.distributed.array.distributed_array` provides a more user
        friendly way of creating a distributed array from another array.

        Args:
            shape (tuple of ints): Length of axes.
            dtype: Data type. It must be an argument of :class:`numpy.dtype`.
            mode (str or mode object): Mode that determines how overlaps of
                chunks are interpreted.
            comms (optional): Communicator objects which a distributed array
                hold internally. Sharing them with other distributed arrays can
                save time because their initialization is a costly operation.
                For details, check
                :meth:`~cupyx.distributed.array.DistributedArray.comms`
                property.
        """
        mem = _MultiDeviceDummyMemory(0)
        memptr = _MultiDeviceDummyPointer(mem, 0)
        obj = super().__new__(cls, shape, dtype, memptr=memptr)
        obj._chunks_map = chunks_map

        obj._streams = {}
        for dev in chunks_map.keys():
            with Device(dev):
                obj._streams[dev] = Stream()

        if isinstance(mode, str):
            if mode not in _MODES.keys():
                raise RuntimeError(
                    f'`mode` must be one of {list(_MODES)}')
            mode = _MODES[str]
        obj._mode = mode

        obj._comms = comms if comms is not None else {}

        return obj

    def __array_finalize__(self, obj):
        # TODO set sensible defualts
        if obj is None:
            return
        self._chunks_map = getattr(obj, '_chunks_map', None)
        self._streams = getattr(obj, '_streams', None)
        self._mode = getattr(obj, '_mode', None)
        self._comms = getattr(obj, '_comms', None)

    @property
    def mode(self) -> str:
        """Describe how overlaps of the chunks are interpreted.

        In the replica mode, chunks are guaranteed to have identical values on
        their overlapping segments. In other modes, they are not necessarily
        identical and represent the original data as their max, sum, etc.

        Some operations on distributed arrays involve changing their mode
        beforehand. For example, see
        :meth:`cupyx.distributed.array.DistributedArray.__matmul__`.
        """
        for mode_str, mode_obj in _MODES.items():
            if self._mode is mode_obj:
                return mode_str
        raise RuntimeError('Unrecognized mode')

    @property
    def devices(self) -> Iterable[int]:
        """A collection of device IDs holding part of the data."""
        return self._chunks_map.keys()

    @property
    def index_map(self) -> dict[int, list[tuple[slice, ...]]]:
        """Indices for the chunks that each device owns."""
        return {dev: [chunk.index for chunk in chunks]
                for dev, chunks in self._chunks_map.items()}

    @property
    def comms(self) -> dict[int, NcclCommunicator]:
        """Communicator objects for data transfer between devices using NCCL.

        They are initialized automatically when needed.

        If NCCL is unavailable, they are replaced by empty mock objects.
        """
        # TODO: Cache in the global scope
        return self._comms

    def _prepare_comms_and_streams(self, devices: Iterable[int]) -> None:
        # Ensure communicators and streams are prepared for communication
        # between `devices` and the devices currently owning chunks
        devices = self._chunks_map.keys() | devices

        if not devices.issubset(self._comms.keys()):
            self._comms = _data_transfer._create_communicators(devices)

        for dev in devices - self._streams.keys():
            with Device(dev):
                self._streams[dev] = Stream()

    def __cupy_override_elementwise_kernel__(self, kernel, *args, **kwargs):
        # This method is called from cupy.ufunc and cupy.ElementwiseKernel
        # to dispatch elementwise operations
        return _elementwise._execute(kernel, args, kwargs)

    def __cupy_override_reduction_kernel__(
            self, kernel, axis, dtype, out, keepdims) -> Any:
        # This method is called from _SimpleReductionKernel and elementary
        # reduction methods of cupy.ndarray to dispatch reduction operations
        # TODO: Support user-defined ReductionKernel
        if axis is None:
            raise RuntimeError('axis must be specified')
        if out is not None:
            raise RuntimeError('Argument `out` is not supported')
        if keepdims:
            raise RuntimeError('Argument `keepdims` is not supported')

        return _reduction._execute(self, kernel, axis, dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc.__name__ == 'matmul' and method == '__call__':
            return linalg._matmul(*inputs, **kwargs)
        return NotImplemented

    def __matmul__(x, y):
        """Matrix multiplication between distributed arrays.

        This operation converts its operands into the replica mode, and compute
        their product in the sum mode."""
        if isinstance(y, DistributedArray):
            return linalg._matmul(x, y)
        else:
            return NotImplemented

    # ------ Mode conversions ------

    def _copy_chunks_map_in_replica_mode(self) -> dict[int, list[_Chunk]]:
        # Return a copy of self.chunks_map in the replica mode
        chunks_map = {}
        for dev, chunks in self._chunks_map.items():
            chunks_map[dev] = [chunk.copy() for chunk in chunks]

        if _is_op_mode(self._mode):
            _chunk._all_reduce_intersections(
                self._mode, self.shape, chunks_map, self._comms, self._streams)

        return chunks_map

    def _copy_chunks_map_in_op_mode(
            self, op_mode: _OpMode) -> dict[int, list[_Chunk]]:
        # Return a copy of self.chunks_map in the given op mode
        chunks_map = self._copy_chunks_map_in_replica_mode()

        for chunk in chain.from_iterable(chunks_map.values()):
            chunk.apply_updates(_REPLICA_MODE)

        chunks_list = list(chain.from_iterable(chunks_map.values()))
        identity = op_mode.identity_of(self.dtype)

        # TODO: Fair distribution of work
        # In the current implementation, devices that appear earlier have to
        # execute set_identity_on_intersection repeatedly, whereas the last
        # device has no work to do
        for i in range(len(chunks_list)):
            a_chunk = chunks_list[i]
            for j in range(i + 1, len(chunks_list)):
                b_chunk = chunks_list[j]
                a_chunk.set_identity_on_intersection(
                    b_chunk.index, self.shape, identity)

        return chunks_map

    def _to_op_mode(self, op_mode: _OpMode) -> 'DistributedArray':
        # Return a view or a copy of the chunks_map in the given mode
        if self._mode is op_mode:
            return self

        if len(self._chunks_map) == 1:
            chunks, = self._chunks_map.values()
            if len(chunks) == 1:
                chunks[0].apply_updates(self._mode)
                return DistributedArray(
                    self.shape, self.dtype, self._chunks_map,
                    op_mode, self._comms)

        chunks_map = self._copy_chunks_map_in_op_mode(op_mode)
        return DistributedArray(
            self.shape, self.dtype, chunks_map, op_mode, self._comms)

    def _to_replica_mode(self) -> 'DistributedArray':
        # Return a view or a copy in the replica mode
        if self._mode is _REPLICA_MODE:
            return self

        if len(self._chunks_map) == 1:
            chunks, = self._chunks_map.values()
            if len(chunks) == 1:
                chunks[0].apply_updates(self._mode)
                return DistributedArray(
                    self.shape, self.dtype, self._chunks_map,
                    _REPLICA_MODE, self._comms)

        chunks_map = self._copy_chunks_map_in_replica_mode()
        return DistributedArray(
            self.shape, self.dtype, chunks_map,
            _REPLICA_MODE, self._comms)

    def change_mode(self, mode: str) -> 'DistributedArray':
        """Return a view or a copy in the given mode."""
        if mode not in _MODES:
            raise RuntimeError(f'`mode` must be one of {list(_MODES)}')

        mode_obj = _MODES[mode]

        if _is_op_mode(mode_obj):
            return self._to_op_mode(mode_obj)
        else:
            return self._to_replica_mode()

    def reshard(self, index_map: dict[int, Any]) -> 'DistributedArray':
        """Return a view or a copy having the given index_map."""
        new_index_map = _index_arith._normalize_index_map(self.shape, index_map)
        if new_index_map == self.index_map:
            return self

        old_chunks_map = self._chunks_map
        new_chunks_map: dict[int, list[_Chunk]] = {}

        # Set up new_chunks_map compatible with new_index_map
        # as placeholders of chunks
        for dev, idxs in new_index_map.items():
            new_chunks_map[dev] = []

            for idx in idxs:
                with Device(dev):
                    dst_shape = _index_arith._shape_after_indexing(
                        self.shape, idx)
                    new_chunk = _Chunk.create_placeholder(dst_shape, dev, idx)
                    new_chunks_map[dev].append(new_chunk)

        self._prepare_comms_and_streams(index_map.keys())

        # Data transfer from old chunks to new chunks
        # TODO: Reorder transfers to minimize latency

        # The current implementation transfers the same data multiple times
        # where chunks overlap. This is particularly problematic when matrix
        # multiplication is involved, where one block tends to be shared between
        # multiple devices
        # TODO: Avoid duplicate data transfers
        for src_chunk in chain.from_iterable(old_chunks_map.values()):
            src_chunk.apply_updates(self._mode)

            if _is_op_mode(self._mode):
                src_chunk = src_chunk.copy()

            for dst_chunk in chain.from_iterable(new_chunks_map.values()):
                src_chunk.apply_to(
                    dst_chunk, self._mode, self.shape,
                    self._comms, self._streams)

        return DistributedArray(
            self.shape, self.dtype, new_chunks_map, self._mode, self._comms)

    def asnumpy(self) -> numpy.ndarray:
        """Return a copy of the array on the host memory."""
        for chunk in chain.from_iterable(self._chunks_map.values()):
            chunk.apply_updates(self._mode)

        if _is_op_mode(self._mode):
            identity = self._mode.identity_of(self.dtype)
            np_array = numpy.full(self.shape, identity, self.dtype)
        else:
            np_array = numpy.empty(self.shape, dtype=self.dtype)

        # We avoid 0D array because we expect data[idx] to return a view
        np_array = numpy.atleast_1d(np_array)

        for chunk in chain.from_iterable(self._chunks_map.values()):
            chunk.ready.synchronize()
            idx = chunk.index
            if _is_op_mode(self._mode):
                self._mode.numpy_func(
                    np_array[idx], cupy.asnumpy(chunk.data), np_array[idx])
            else:
                np_array[idx] = cupy.asnumpy(chunk.data)

        # Undo numpy.atleast_1d
        return np_array.reshape(self.shape)


def distributed_array(
    array: ArrayLike,
    index_map: dict[int, Any],
    mode: str = 'replica',
    comms: Optional[dict[int, NcclCommunicator]] = None,
) -> DistributedArray:
    """Creates a distributed array from the given data.

    This function does not check if all elements of ``array`` are stored in some
    of the chunks.

    Args:
        array: :class:`~cupyx.distributed.array.DistributedArray` object,
            :class:`cupy.ndarray` object or any other object that can be passed
            to :func:`numpy.array`.
        index_map (dict from int to array indices): Indices for the chunks
            that devices with designated IDs own. One device can have multiple
            chunks, which can be specified as a list of array indices.
        comms (optional): Communicator objects which a distributed array hold
            internally. Sharing them with other distributed arrays can save time
            because their initialization is a costly operation. For details,
            check :meth:`~cupyx.distributed.array.DistributedArray.comms`
            property.

    Example:
        >>> array = cupy.arange(9).reshape(3, 3)
        >>> A = distributed_array(
        ...     array,
        ...     {0: [(slice(2), slice(2)),  # array[:2, :2]
        ...          slice(None, None, 2)], # array[::2]
        ...      1:  (slice(1, None), 2)})  # array[1:, 2]
    """
    if isinstance(array, DistributedArray):
        if array.mode != mode:
            array = array.change_mode(mode)
        if array.index_map != index_map:
            array = array.reshard(index_map)
        return DistributedArray(
            array.shape, array.dtype, array._chunks_map, array._mode, comms)

    if isinstance(array, (numpy.ndarray, cupy.ndarray)):
        if mode != 'replica':
            array = array.copy()
    else:
        array = numpy.array(array)

    new_index_map = _index_arith._normalize_index_map(array.shape, index_map)

    # Define how to create chunks from the original data
    if isinstance(array, cupy.ndarray):
        src_dev = array.device.id
        with Device(src_dev):
            src_stream = get_current_stream()

        devices = index_map.keys() | {array.device.id}
        if comms is None or not devices.issubset(comms.keys()):
            comms = _data_transfer._create_communicators(devices)

        def make_chunk(dst_dev, idx, src_array):
            with src_array.device:
                src_array = cupy.ascontiguousarray(src_array)
                src_data = _data_transfer._AsyncData(
                    src_array, src_stream.record(), prevent_gc=src_array)
            with Device(dst_dev):
                dst_stream = get_current_stream()
                copied = _data_transfer._transfer(
                    comms[src_dev], src_stream, src_data,
                    comms[dst_dev], dst_stream, dst_dev)
                return _Chunk(copied.data, copied.ready, idx,
                              prevent_gc=src_data)
    else:
        def make_chunk(dev, idx, array):
            with Device(dev):
                stream = get_current_stream()
                copied = cupy.array(array)
                return _Chunk(copied, stream.record(), idx,
                              prevent_gc=array)

    mode_obj = _MODES[mode]
    chunks_map: dict[int, list[_Chunk]] = {}
    for dev, idxs in new_index_map.items():
        chunks_map[dev] = []

        for i, idx in enumerate(idxs):
            chunk_data = array[idx]
            chunk = make_chunk(dev, idx, chunk_data)
            chunks_map[dev].append(chunk)
            if _is_op_mode(mode_obj) and not mode_obj.idempotent:
                array[idx] = mode_obj.identity_of(array.dtype)

    return DistributedArray(
        array.shape, array.dtype, chunks_map, mode_obj, comms)
