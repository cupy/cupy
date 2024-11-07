from itertools import chain
from typing import Any, Callable, Iterable, Optional

import numpy
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike

import cupy
from cupy._core.core import ndarray
import cupy._creation.from_data as _creation_from_data
import cupy._core._routines_math as _math
import cupy._core._routines_statistics as _statistics
from cupy.cuda.device import Device
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream

from cupyx.distributed.array import _chunk
from cupyx.distributed.array._chunk import _Chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array._data_transfer import _Communicator
from cupyx.distributed.array import _elementwise
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _modes
from cupyx.distributed.array import _reduction
from cupyx.distributed.array import _linalg


class _MultiDeviceDummyMemory(cupy.cuda.memory.Memory):
    pass


class _MultiDeviceDummyPointer(cupy.cuda.memory.MemoryPointer):
    @property
    def device(self) -> Device:
        # This override is needed to assign an invalid device id
        # Since the array is not residing in a single device now
        return Device(-1)


def _make_chunk_async(src_dev, dst_dev, idx, src_array, comms):
    src_stream = get_current_stream(src_dev)
    with src_array.device:
        src_array = _creation_from_data.ascontiguousarray(src_array)
        src_data = _data_transfer._AsyncData(
            src_array, src_stream.record(), prevent_gc=src_array)
    with Device(dst_dev):
        dst_stream = get_current_stream()
        copied = _data_transfer._transfer(
            comms[src_dev], src_stream, src_data,
            comms[dst_dev], dst_stream, dst_dev)
        return _Chunk(copied.array, copied.ready, idx,
                      prevent_gc=src_data)


def _make_chunk_sync(src_dev, dst_dev, idx, src_array, comms):
    with Device(dst_dev):
        stream = get_current_stream()
        copied = _creation_from_data.array(src_array)
        return _Chunk(copied, stream.record(), idx, prevent_gc=src_array)


class DistributedArray(ndarray):
    """
    __init__(self, shape, dtype, chunks_map, mode=REPLICA, comms=None)

    Multi-dimensional array distributed across multiple CUDA devices.

    This class implements some elementary operations that :class:`cupy.ndarray`
    provides. The array content is split into chunks, contiguous arrays
    corresponding to slices of the original array. Note that one device can
    hold multiple chunks.

    This direct constructor is designed for internal calls. Users should create
    distributed arrays using :func:`distributed_array`.

    Args:
        shape (tuple of ints): Shape of created array.
        dtype (dtype_like): Any object that can be interpreted as a numpy data
            type.
        chunks_map (dict from int to list of chunks): Lists of chunk objects
            associated with each device.
        mode (mode object, optional): Mode that determines how overlaps
            of the chunks are interpreted. Defaults to
            ``cupyx.distributed.array.REPLICA``.
        comms (optional): Communicator objects which a distributed array
            hold internally. Sharing them with other distributed arrays can
            save time because their initialization is a costly operation.

    .. seealso::
            :attr:`DistributedArray.mode` for details about modes.
    """

    _chunks_map: dict[int, list[_Chunk]]
    _mode: _modes.Mode
    _streams: dict[int, Stream]
    _comms: dict[int, _Communicator]

    def __new__(
        cls, shape: tuple[int, ...], dtype: DTypeLike,
        chunks_map: dict[int, list[_Chunk]],
        mode: _modes.Mode = _modes.REPLICA,
        comms: Optional[dict[int, _Communicator]] = None,
    ) -> 'DistributedArray':
        mem = _MultiDeviceDummyMemory(0)
        memptr = _MultiDeviceDummyPointer(mem, 0)
        obj = super().__new__(cls, shape, dtype, memptr=memptr)
        obj._chunks_map = chunks_map

        obj._mode = mode

        obj._streams = {}
        obj._comms = comms if comms is not None else {}

        return obj

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __array_finalize__(self, obj):
        if obj is not None:
            raise RuntimeError(
                'Distributed array can only be instantiated by an explicit'
                'constructor call')

    @property
    def mode(self) -> _modes.Mode:
        """Describe how overlaps of the chunks are interpreted.

        In the replica mode, chunks are guaranteed to have identical values on
        their overlapping segments. In other modes, they are not necessarily
        identical and represent the original data as their max, sum, etc.

        :class:`DistributedArray` currently supports
        ``cupyx.distributed.array.REPLICA``, ``cupyx.distributed.array.MIN``,
        ``cupyx.distributed.array.MAX``, ``cupyx.distributed.array.SUM``,
        ``cupyx.distributed.array.PROD`` modes.

        Many operations on distributed arrays including :class:`cupy.ufunc`
        and :func:`~cupyx.distributed.array.matmul` involve changing their mode
        beforehand. These mode conversions are done automatically, so in most
        cases users do not have to manage modes manually.

        Example:
            >>> A = distributed_array(
            ...     cupy.arange(6).reshape(2, 3),
            ...     make_2d_index_map([0, 2], [0, 1, 3],
            ...                       [[{0}, {1, 2}]]))
            >>> B = distributed_array(
            ...     cupy.arange(12).reshape(3, 4),
            ...     make_2d_index_map([0, 1, 3], [0, 2, 4],
            ...                       [[{0}, {0}],
            ...                        [{1}, {2}]]))
            >>> C = A @ B
            >>> C
            array([[20, 23, 26, 29],
                   [56, 68, 80, 92]])
            >>> C.mode
            'sum'
            >>> C.all_chunks()
            {0: [array([[0, 0],
                        [0, 3]]),     # left half
                 array([[0, 0],
                        [6, 9]])],    # right half
             1: [array([[20, 23],
                        [56, 65]])],  # left half
             2: [array([[26, 29],
                        [74, 83]])]}  # right half
            >>> C_replica = C.change_mode('replica')
            >>> C_replica.mode
            'replica'
            >>> C_replica.all_chunks()
            {0: [array([[20, 23],
                        [56, 68]]),   # left half
                 array([[26, 29],
                        [80, 92]])],  # right half
             1: [array([[20, 23],
                        [56, 68]])],  # left half
             2: [array([[26, 29],
                        [80, 92]])]}  # right half
        """
        return self._mode

    @property
    def devices(self) -> Iterable[int]:
        """A collection of device IDs holding part of the data."""
        return self._chunks_map.keys()

    @property
    def index_map(self) -> dict[int, list[tuple[slice, ...]]]:
        """Indices for the chunks that devices with designated IDs own."""
        return {dev: [chunk.index for chunk in chunks]
                for dev, chunks in self._chunks_map.items()}

    def all_chunks(self) -> dict[int, list[ndarray]]:
        """Return the chunks with all buffered data flushed.

        Buffered data are created in situations such as resharding and mode
        changing.
        """
        chunks_map: dict[int, list[ndarray]] = {}
        for dev, chunks in self._chunks_map.items():
            chunks_map[dev] = []
            for chunk in chunks:
                chunk.flush(self._mode)
                chunks_map[dev].append(chunk.array)
        return chunks_map

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
        # reduction methods of ndarray to dispatch reduction operations
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
            return _linalg.matmul(*inputs, **kwargs)
        return NotImplemented

    def __matmul__(x, y):
        if isinstance(y, DistributedArray):
            return _linalg.matmul(x, y)
        else:
            return NotImplemented

    def _copy_chunks_map_in_replica_mode(self) -> dict[int, list[_Chunk]]:
        # Return a copy of self.chunks_map in the replica mode
        chunks_map = {}
        for dev, chunks in self._chunks_map.items():
            chunks_map[dev] = [chunk.copy() for chunk in chunks]

        if self._mode is not _modes.REPLICA:
            self._prepare_comms_and_streams(self._chunks_map.keys())
            _chunk._all_reduce_intersections(
                self._mode, self.shape, chunks_map, self._comms, self._streams)

        return chunks_map

    def _copy_chunks_map_in_op_mode(
            self, op_mode: _modes._OpMode) -> dict[int, list[_Chunk]]:
        # Return a copy of self.chunks_map in the given op mode
        chunks_map = self._copy_chunks_map_in_replica_mode()

        for chunk in chain.from_iterable(chunks_map.values()):
            chunk.flush(_modes.REPLICA)

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

    def _to_op_mode(self, op_mode: _modes.Mode) -> 'DistributedArray':
        # Return a view or a copy of the chunks_map in the given mode
        if self._mode is op_mode:
            return self

        if len(self._chunks_map) == 1:
            chunks, = self._chunks_map.values()
            if len(chunks) == 1:
                chunks[0].flush(self._mode)
                return DistributedArray(
                    self.shape, self.dtype, self._chunks_map,
                    op_mode, self._comms)
        if op_mode is _modes.REPLICA:
            chunks_map = self._copy_chunks_map_in_replica_mode()
        else:
            assert op_mode is not None
            chunks_map = self._copy_chunks_map_in_op_mode(op_mode)
        return DistributedArray(
            self.shape, self.dtype, chunks_map, op_mode, self._comms)

    def change_mode(self, mode: _modes.Mode) -> 'DistributedArray':
        """Return a view or a copy in the given mode.

        Args:
            mode (mode Object): How overlaps of
                the chunks are interpreted.

        .. seealso::
                :attr:`DistributedArray.mode` for details about modes.
        """
        return self._to_op_mode(mode)

    def reshard(self, index_map: dict[int, Any]) -> 'DistributedArray':
        """Return a view or a copy having the given index_map.

        Data transfers across devices are done on separate streams created
        internally. To make them asynchronous, transferred data is buffered and
        reflected to the chunks when necessary.

        Args:
            index_map (dict from int to array indices): Indices for the chunks
                that devices with designated IDs own. The current index_map of
                a distributed array can be obtained from
                :attr:`DistributedArray.index_map`.
        """
        new_index_map = _index_arith._normalize_index_map(
            self.shape, index_map)
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
        # multiplication is involved, where one block tends to be shared
        # between multiple devices
        # TODO: Avoid duplicate data transfers
        for src_chunk in chain.from_iterable(old_chunks_map.values()):
            src_chunk.flush(self._mode)

            if self._mode is not _modes.REPLICA:
                src_chunk = src_chunk.copy()

            for dst_chunk in chain.from_iterable(new_chunks_map.values()):
                src_chunk.apply_to(
                    dst_chunk, self._mode, self.shape,
                    self._comms, self._streams)

        return DistributedArray(
            self.shape, self.dtype, new_chunks_map, self._mode, self._comms)

    def get(
        self, stream=None, order='C', out=None, blocking=True
    ) -> numpy.ndarray:
        """Return a copy of the array on the host memory."""
        if stream is not None:
            raise RuntimeError('Argument `stream` not supported')
        if order != 'C':
            raise RuntimeError('Argument `order` not supported')
        if out is not None:
            raise RuntimeError('Argument `out` not supported')

        for chunk in chain.from_iterable(self._chunks_map.values()):
            chunk.flush(self._mode)

        if self._mode is _modes.REPLICA:
            np_array = numpy.empty(self.shape, dtype=self.dtype)
        else:
            identity = self._mode.identity_of(self.dtype)
            np_array = numpy.full(self.shape, identity, self.dtype)

        # We avoid 0D array because we expect data[idx] to return a view
        np_array = numpy.atleast_1d(np_array)

        for chunk in chain.from_iterable(self._chunks_map.values()):
            chunk.ready.synchronize()
            idx = chunk.index
            if self._mode is _modes.REPLICA:
                np_array[idx] = cupy.asnumpy(chunk.array)
            else:
                self._mode.numpy_func(
                    np_array[idx], cupy.asnumpy(chunk.array), np_array[idx])

        # Undo numpy.atleast_1d
        return np_array.reshape(self.shape)

    # -----------------------------------------------------
    # Overriding unsupported methods inherited from ndarray
    # -----------------------------------------------------

    def __getitem__(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support __getitem__.')

    def __setitem__(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support __setitem__.')

    def __len__(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support __len__.')

    def __iter__(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support __iter__.')

    def __copy__(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support __copy__.')

    def all(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support all.')

    def any(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support any.')

    def argmax(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support argmax.')

    def argmin(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support argmin.')

    def argpartition(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support argpartition.')

    def argsort(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support argsort.')

    def astype(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support astype.')

    def choose(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support choose.')

    def clip(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support clip.')

    def compress(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support compress.')

    def copy(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support copy.')

    def cumprod(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support cumprod.')

    def cumsum(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support cumsum.')

    def diagonal(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support diagonal.')

    def dot(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support dot.')

    def dump(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support dump.')

    def dumps(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support dumps.')

    def fill(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support fill.')

    def flatten(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support flatten.')

    def item(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support item.')

    def max(self, axis=None, out=None, keepdims=False):
        """Return the maximum along a given axis.

        .. note::

            Currently, it only supports non-``None`` values for ``axis`` and
            the default values for ``out`` and ``keepdims``.

        .. seealso::
           :meth:`cupy.ndarray.max`, :meth:`numpy.ndarray.max`
        """
        return self.__cupy_override_reduction_kernel__(
            _statistics.amax, axis, None, out, keepdims)

    def mean(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support mean.')

    def min(self, axis=None, out=None, keepdims=False):
        """Return the minimum along a given axis.

        .. note::

            Currently, it only supports non-``None`` values for ``axis`` and
            the default values for ``out`` and ``keepdims``.

        .. seealso::
           :meth:`cupy.ndarray.min`, :meth:`numpy.ndarray.min`
        """
        return self.__cupy_override_reduction_kernel__(
            _statistics.amin, axis, None, out, keepdims)

    def nonzero(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support nonzero.')

    def partition(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support partition.')

    def prod(self, axis=None, dtype=None, out=None, keepdims=None):
        """Return the minimum along a given axis.

        .. note::

            Currently, it only supports non-``None`` values for ``axis`` and
            the default values for ``out`` and ``keepdims``.

        .. seealso::
           :meth:`cupy.ndarray.prod`, :meth:`numpy.ndarray.prod`
        """
        if dtype is None:
            return self.__cupy_override_reduction_kernel__(
                _math.prod_auto_dtype, axis, dtype, out, keepdims)
        else:
            return self.__cupy_override_reduction_kernel__(
                _math.prod_keep_dtype, axis, dtype, out, keepdims)

    def ptp(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support ptp.')

    def put(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support put.')

    def ravel(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support ravel.')

    def reduced_view(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support reduced_view.')

    def repeat(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support repeat.')

    def reshape(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support reshape.')

    def round(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support round.')

    def scatter_add(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support scatter_add.')

    def scatter_max(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support scatter_max.')

    def scatter_min(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support scatter_min.')

    def searchsorted(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support searchsorted.')

    def set(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support set.')

    def sort(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support sort.')

    def squeeze(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support squeeze.')

    def std(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support std.')

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """Return the minimum along a given axis.

        .. note::

            Currently, it only supports non-``None`` values for ``axis`` and
            the default values for ``out`` and ``keepdims``.

        .. seealso::
           :meth:`cupy.ndarray.sum`, :meth:`numpy.ndarray.sum`
        """
        if dtype is None:
            return self.__cupy_override_reduction_kernel__(
                _math.sum_auto_dtype, axis, dtype, out, keepdims)
        else:
            return self.__cupy_override_reduction_kernel__(
                _math.sum_keep_dtype, axis, dtype, out, keepdims)

    def swapaxes(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support swapaxes.')

    def take(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support take.')

    def toDlpack(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support toDlpack.')

    def tobytes(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support tobytes.')

    def tofile(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support tofile.')

    def tolist(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support tolist.')

    def trace(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support trace.')

    def transpose(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support transpose.')

    def var(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support var.')

    def view(self, *args, **kwargs):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support view.')

    @property
    def T(self):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support T.')

    @property
    def base(self):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support base.')

    @property
    def cstruct(self):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support cstruct.')

    @property
    def data(self):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support data.')

    @property
    def device(self):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support device.')

    @property
    def flags(self):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support flags.')

    @property
    def flat(self):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support flat.')

    @property
    def imag(self):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support imag.')

    @property
    def real(self):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support real.')

    @property
    def shape(self):
        """Tuple of array dimensions.

        Assignment to this property is currently not supported.

        .. seealso: :attr:`cupy.ndarray.shape`, :attr:`numpy.ndarray.shape`

        """
        return super().shape

    @shape.setter
    def shape(self, newshape):
        raise NotImplementedError(
            'DistributedArray currently does not support assignment to shape.')

    @property
    def strides(self):
        """Not supported."""
        raise NotImplementedError(
            'DistributedArray currently does not support strides.')


def distributed_array(
    array: ArrayLike,
    index_map: dict[int, Any],
    mode: _modes.Mode = _modes.REPLICA,
) -> DistributedArray:
    """Creates a distributed array from the given data.

    This function does not check if all elements of the given array are stored
    in some of the chunks.

    Args:
        array (array_like): :class:`DistributedArray` object,
            :class:`cupy.ndarray` object or any other object that can be passed
            to :func:`numpy.array`.
        index_map (dict from int to array indices): Indices for the chunks
            that devices with designated IDs own. One device can have multiple
            chunks, which can be specified as a list of array indices.
        mode (mode object, optional): Mode that determines how overlaps
            of the chunks are interpreted. Defaults to
            ``cupyx.distributed.array.REPLICA``.

    .. seealso::
            :attr:`DistributedArray.mode` for details about modes.

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
            array.shape, array.dtype, array._chunks_map, array._mode,
            array._comms)

    if isinstance(array, (numpy.ndarray, ndarray)):
        if mode != _modes.REPLICA:
            array = array.copy()
    else:
        array = numpy.array(array)

    index_map = _index_arith._normalize_index_map(array.shape, index_map)
    comms = None

    # Define how to form a chunk from (dev, idx, src_array)
    make_chunk: Callable[
        [int, int, tuple[slice, ...], ndarray, Optional[list[Any]]],
        _Chunk
    ]

    if isinstance(array, ndarray):
        src_dev = array.device.id
        devices = index_map.keys() | {array.device.id}
        comms = _data_transfer._create_communicators(devices)
        make_chunk = _make_chunk_async
    else:
        src_dev = -1
        make_chunk = _make_chunk_sync

    chunks_map: dict[int, list[_Chunk]] = {}
    for dev, idxs in index_map.items():
        chunks_map[dev] = []

        for idx in idxs:
            chunk_array = array[idx]
            chunk = make_chunk(src_dev, dev, idx, chunk_array, comms)
            chunks_map[dev].append(chunk)
            if (mode is not _modes.REPLICA
                    and not mode.idempotent):
                array[idx] = mode.identity_of(array.dtype)

    return DistributedArray(
        array.shape, array.dtype, chunks_map, mode, comms)
