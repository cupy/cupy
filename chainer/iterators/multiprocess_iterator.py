from __future__ import division
import multiprocessing
from multiprocessing import sharedctypes
import threading
import warnings

import numpy
import six

from chainer.dataset import iterator


class MultiprocessIterator(iterator.Iterator):

    """Dataset iterator that loads examples in parallel.

    This is an implementation of :class:`~chainer.dataset.Iterator` that loads
    examples with worker processes. It uses the standard :mod:`multiprocessing`
    module to parallelize the loading. The dataset is sent to the worker
    processes in the standard way using pickle.

    Note that this iterator effectively prefetches the examples for the next
    batch asynchronously after the current batch is returned.

    Args:
        dataset (~chainer.dataset.Dataset): Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes.
        n_processes (int): Number of worker processes. The number of CPUs is
            used by default.

    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True,
                 n_processes=None, n_prefetch=1, sharedmem_size=1000000):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        if shuffle:
            self._order = numpy.random.permutation(len(dataset))
        else:
            self._order = None
        self._prefetch_order = None  # used at the end of each epoch

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self._pushed_position = None  # initialized at the first iteration

        self.n_processes = n_processes or multiprocessing.cpu_count()
        self.n_prefetch = n_prefetch
        self.sharedmem_size = sharedmem_size

        self._start = False
        self._finalized = False

    def __del__(self):
        self.finalize()

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self.is_new_epoch = False
        if not self._start:
            self._init()  # start workers
            # load for the first iteration
            for _ in six.moves.range(self.n_prefetch):
                self._invoke_prefetch()

        batch = self._get()
        self._invoke_prefetch()  # prefetch for the next iteration
        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    def finalize(self):
        if not self._start or self._finalized:
            return

        self._finalized = True
        try:
            while True:
                self._data_queue.get_nowait()
        except six.moves.queue.Empty:
            pass
        for _ in self._workers:
            self._index_queue.put((-1, -1, -1))  # termination signal
        for worker in self._workers:
            worker.join()

        self._get_data_loop_thread.join()

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        serializer('order', self._order)

    def _init(self):
        self._index_queue = multiprocessing.Queue()
        self._data_queue = multiprocessing.Queue()
        self._orderd_data_queue = six.moves.queue.Queue()
        self._unused_sharedmem_queue = six.moves.queue.Queue()

        self._sharedmem_list = [
            sharedctypes.RawArray('b', self.sharedmem_size)
            for _ in six.moves.range(self.batch_size * self.n_prefetch)]
        for i in six.moves.range(len(self._sharedmem_list)):
            self._unused_sharedmem_queue.put(i)
        self._cnt = 0

        args = (self.dataset, self._index_queue, self._data_queue,
                self._sharedmem_list)
        self._workers = []
        for _ in range(self.n_processes):
            worker = multiprocessing.Process(target=_worker, args=args)
            worker.daemon = True
            self._workers.append(worker)
            worker.start()
        self._get_data_loop_thread = threading.Thread(
            target=self._get_data_loop, name="get_data_loop")
        self._get_data_loop_thread.daemon = True
        self._get_data_loop_thread.start()

        self._start = True

    def _invoke_prefetch(self):
        N = len(self.dataset)
        i = self._pushed_position
        if i is None:  # first iteration
            i = self.current_position

        order = self._order
        for k in six.moves.range(self.batch_size):
            if i >= N:
                if not self._repeat:
                    break
                i = 0
                if order is not None:
                    # We cannot shuffle the order directly here, since the
                    # iterator may be serialized before the prefetched data are
                    # consumed by the user, in which case an inconsistency
                    # appears.
                    order = order.copy()
                    numpy.random.shuffle(order)
            index = i if order is None else order[i]
            self._index_queue.put(
                (self._cnt, self._unused_sharedmem_queue.get(), index))
            self._cnt += 1
            i += 1

        self._prefetch_order = order  # Temporarily store the shuffled order.
        self._pushed_position = i

    def _get_data_loop(self):
        buf = dict()
        cnt = 0
        while not self._finalized:
            if cnt in buf:
                data = buf.pop(cnt)
            else:
                try:
                    c, mem_index, data = self._data_queue.get(timeout=1)
                except six.moves.queue.Empty:
                    continue
                data = _unpack(data, self._sharedmem_list[mem_index])
                self._unused_sharedmem_queue.put(mem_index)
                if c != cnt:
                    buf[c] = data
                    continue
            self._orderd_data_queue.put(data)
            del data
            cnt += 1

    def _get(self):
        N = len(self.dataset)
        i = self.current_position

        batch = []
        for k in six.moves.range(self.batch_size):
            batch.append(self._orderd_data_queue.get())
            i += 1
            if i >= N:
                self.epoch += 1
                self.is_new_epoch = True
                if not self._repeat:
                    break
                i = 0
        self.current_position = i
        # Eventually overwrite the (possibly shuffled) order.
        self._order = self._prefetch_order
        return batch


class _PackedNdarray(object):

    def __init__(self, array, mem, offset):
        self.shape = array.shape
        self.dtype = array.dtype
        self.nbytes = array.nbytes
        self.size = array.size
        self.offset = offset
        total = self.offset + self.nbytes
        if total > len(mem):
            raise ValueError(
                'Shared memory size is too small. expect:{}, actual:{}'.format(
                    total, len(mem)))
        target = numpy.frombuffer(mem, self.dtype, self.size, self.offset)
        target[...] = array.ravel()

    def unpack(self, mem):
        ret = numpy.frombuffer(mem, self.dtype, self.size, self.offset)
        ret = ret.reshape(self.shape).copy()
        return ret


def _pack(data, mem):
    if len(mem) == 0:
        return data
    t = type(data)
    if t is tuple or t is list:
        ret = []
        offset = 0
        expect = 0
        for v in data:
            if isinstance(v, numpy.ndarray):
                expect += v.nbytes
                if v.nbytes + offset <= len(mem):
                    v = _PackedNdarray(v, mem, offset)
                    offset += v.nbytes

            ret.append(v)
        if expect > len(mem):
            warnings.warn(
                'Shared memory size is too small. expect:{}, actual:{}'.format(
                    expect, len(mem)), UserWarning)
        data = t(ret)
    return data


def _unpack(data, mem):
    if len(mem) == 0:
        return data
    t = type(data)
    if t is tuple or t is list:
        ret = []
        for v in data:
            if isinstance(v, _PackedNdarray):
                v = v.unpack(mem)
            ret.append(v)
        data = t(ret)
    return data


def _worker(dataset, in_queue, out_queue, sharedmem_list):
    while True:
        cnt, mem_index, index = in_queue.get()
        if cnt < 0:
            break
        mem = sharedmem_list[mem_index]
        data = _pack(dataset[index], mem)
        out_queue.put((cnt, mem_index, data))
    out_queue.close()
    out_queue.cancel_join_thread()
