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
            Otherwise, it stops iteration at the end of the sweep.

    """
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._order = numpy.random.permutation(len(dataset))

        self.current_position = 0
        self.epoch = 0

    def next(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        N = len(self.dataset)

        batch = []
        for _ in six.moves.range(batch_size):
            batch.append(self.dataset[self._order[i]])
            i += 1
            if i >= N:
                self.epoch += 1
                i = 0
                if not self._repeat:
                    break
                numpy.random.shuffle(self._order)
        self.current_position = i

        return batch

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        serializer('_order', self._order)
