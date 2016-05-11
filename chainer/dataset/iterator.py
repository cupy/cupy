class Iterator(object):

    """Base class of all dataset iterators.

    Iterator iterates over the dataset, yielding a minibatch at each
    iteration. Minibatch is a list of examples. Each implementation should
    implement an iterator protocol (e.g., the :meth:`next` method).

    Note that, even if the iterator supports setting the batch size, it does
    not guarantee that each batch always contains the same number of examples.
    For example, if you let the iterator to stop at the end of the sweep, the
    last batch may contain a fewer number of examples.

    The interface between the iterator and the underlying dataset is not fixed,
    and up to the implementation.

    Each implementation should also support serialization to resume/suspend the
    iteration.

    """
    def __iter__(self):
        """Returns self."""
        return self

    @property
    def batch_size(self):
        """The number of examples within each batch.

        Implementations should override it or provide an attribute of the same
        name.

        """
        raise NotImplementedError

    @property
    def current_position(self):
        """The current position of the iterator within the epoch.

        It represents how many examples are already visited in the current
        epoch.

        Implementation should override it or provide an attribute of the same
        name.

        """
        raise NotImplementedError

    @property
    def epoch(self):
        """The number of completed sweeps over the dataset.

        The definition of epoch depends on the dataset and iterator
        implementation. In most cases, it represents how many times all data
        points are already visited.

        Implementations should override it or provide an attribute of the same
        name.

        """
        raise NotImplementedError

    @property
    def is_new_epoch(self):
        """True if the epoch count was incremented at the last update."""
        raise NotImplementedError

    def next(self):
        """Returns the next batch.

        This is a part of the iterator protocol of Python. It may raises the
        :class:`StopIteration` exception when it stops the iteration.

        """
        raise NotImplementedError

    def finalize(self):
        """Finalizes the iterator and possibly releases the resources.

        This method does nothing by default. Implementation may override it to
        better handle the internal resources.

        """
        pass

    def serialize(self, serializer):
        """Serializes the internal state of the iterator.

        This is a method to support serializer protocol of Chainer.

        .. note::
           It should only serializes the internal state that changes over the
           iteration. It should not serializes what is set manually by
           users such as the batch size.

        """
        pass
