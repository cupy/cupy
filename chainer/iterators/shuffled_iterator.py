from __future__ import division

import numpy
import six

from chainer.dataset import iterator


class ShuffledIterator(iterator.Iterator):

    """Dataset iterator that yields the examples in a shuffled order.

    This is an implementation of :class:`~chainer.dataset.Iterator` that
    iterates the examples in a shuffled order. The order is generated at each
    epoch.

    Args:
        dataset (~chainer.dataset.Dataset): Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If True, then it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.

    """
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._order = numpy.random.permutation(len(dataset))

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self.is_new_epoch = False

        i = self.current_position
        N = len(self.dataset)

        batch = []
        for _ in six.moves.range(self.batch_size):
            batch.append(self.dataset[self._order[i]])
            i += 1
            if i >= N:
                self.epoch += 1
                self.is_new_epoch = True
                if not self._repeat:
                    i = N
                    break
                i = 0
                numpy.random.shuffle(self._order)
        self.current_position = i

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        serializer('_order', self._order)
