from chainer.dataset import iterator


class SequentialIterator(iterator.Iterator):

    """Dataset iterator that sequentailly yields the examples.

    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in the order of indexes. In general, this
    should not be used for training, but it is useful for validation. It is
    also useful when the order of examples is important and should not be
    broken.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If True, then it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the sweep.

    """
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat

        self.current_position = 0
        self.epoch = 0

    def next(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)

        if i_end >= N:
            batch = self.dataset[i:]
            if self._repeat:
                rest = i_end - N
                if rest > 0:
                    batch = list(batch) + list(self.dataset[:rest])
                self.current_position = rest
            else:
                self.current_position = 0
            self.epoch += 1
        else:
            batch = self.dataset[i:i_end]
            self.current_position = i_end

        return batch

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
