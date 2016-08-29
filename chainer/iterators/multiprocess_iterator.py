from __future__ import division
import multiprocessing

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
                 n_processes=None):
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
            self._invoke_prefetch()  # load for the first iteration
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
            self._index_queue.put(-1)  # termination signal
        for worker in self._workers:
            worker.join()

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        serializer('order', self._order)

    def _init(self):
        queue_size = max(self.n_processes, self.batch_size)
        self._index_queue = multiprocessing.Queue(queue_size)
        self._data_queue = multiprocessing.Queue(queue_size)

        args = self.dataset, self._index_queue, self._data_queue
        self._workers = []
        for _ in range(self.n_processes):
            worker = multiprocessing.Process(target=_worker, args=args)
            worker.daemon = True
            self._workers.append(worker)
            worker.start()
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
            self._index_queue.put(index)
            i += 1

        self._prefetch_order = order  # Temporarily store the shuffled order.
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
                    break
                i = 0
        self.current_position = i
        # Eventually overwrite the (possibly shuffled) order.
        self._order = self._prefetch_order
        return batch


def _worker(dataset, in_queue, out_queue):
    while True:
        index = in_queue.get()
        if index < 0:
            break
        out_queue.put(dataset[index])
    out_queue.close()
    out_queue.cancel_join_thread()
