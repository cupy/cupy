from __future__ import division
import multiprocessing

import numpy
import six

from chainer.dataset import iterator


class MultiprocessIterator(iterator.Iterator):

    """Dataset iterator that loads examples in parallel.

    This is an implementation of :class:`~chainer.dataset.Iterator` that loads
    examples with worker processes. It uses the standard :mod:`multiprocessing`
    module to parallelize the loading. In order to access the dataset by worker
    processes, the dataset is sent to them in the standard way using pickle.

    Note that this iterator effectively prefetches the examples for the next
    batch asynchronously after the current batch is returned.

    This iterator shuffles the order of examples at the beginning of each epoch
    like :class:`ShuffledIterator`.

    Args:
        dataset (~chainer.dataset.Dataset): Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If True, then it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the sweep.
        n_processes (int): Number of worker processes. The number of CPUs is
            used by default.

    """
    def __init__(self, dataset, batch_size, repeat=True, n_processes=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._order = numpy.random.permutation(len(dataset))
        self._prefetch_order = None  # used at the end of each epoch

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self._pushed_position = None  # initialized at the first iteration

        if n_processes is None:
            n_processes = multiprocessing.cpu_count()

        queue_size = max(n_processes, batch_size)
        self._index_queue = multiprocessing.Queue(queue_size)
        self._data_queue = multiprocessing.Queue(queue_size)

        args = dataset, self._index_queue, self._data_queue
        self._workers = []
        for _ in range(n_processes):
            worker = multiprocessing.Process(target=_worker, args=args)
            worker.daemon = True
            self._workers.append(worker)
            worker.start()

        self._start = False
        self._finalized = False

    def __del__(self):
        self.finalize()

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self.is_new_epoch = False
        if not self._start:
            self._invoke_prefetch()  # load for the first iteration
            self._start = True
        batch = self._get()
        self._invoke_prefetch()  # prefetch for the next iteration
        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    @property
    def n_processes(self):
        return len(self._workers)

    def finalize(self):
        if self._finalized:
            return

        self._finalized = True
        workers = self._workers
        self._workers = []
        try:
            while True:
                self._data_queue.get_nowait()
        except six.moves.queue.Empty:
            pass
        for _ in workers:
            self._index_queue.put(-1)  # termination signal
        for worker in workers:
            worker.join()

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)

    def _invoke_prefetch(self):
        N = len(self.dataset)
        i = self._pushed_position
        if i is None:  # first iteration
            i = self.current_position

        for k in six.moves.range(self.batch_size):
            self._index_queue.put(self._order[i])
            i += 1
            if i >= N:
                i = 0
                if not self._repeat:
                    break
                # TODO(beam2d): This shuffle may break the state on suspend.
                # Consider the case that the following occurs in this order:
                #   1. Start prefetching the next batch.
                #   2. Shuffle the order during the prefetch.
                #   3. Iterator is suspended (i.e., serialized).
                #   4. Iterator is resumed (i.e., deserialized) by another
                #      object.
                # In this case, the serialized iterator holds a new (shuffled)
                # order information, while the resumed iterator must start from
                # the end of the epoch with the old order.
                numpy.random.shuffle(self._order)
        self._pushed_position = i

    def _get(self):
        N = len(self.dataset)
        i = self.current_position
        batch = []
        for k in six.moves.range(self.batch_size):
            batch.append(self._data_queue.get())
            i += 1
            if i >= N:
                self.epoch += 1
                self.is_new_epoch = True
                if not self._repeat:
                    i = N
                    break
                i = 0
        self.current_position = i
        return batch


def _worker(dataset, in_queue, out_queue):
    while True:
        index = in_queue.get()
        if index < 0:
            break
        out_queue.put(dataset[index])
