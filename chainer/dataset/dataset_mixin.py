class DatasetMixin(object):

    """Default implementation of dataset indexing.

    DatasetMixin provides the :meth:`__getitem__` method using the per-example
    addressing method :meth:`get_example` given by the implementation. This
    mixin makes it easy to implement a new dataset that does not support
    efficient slicing.

    """
    def __getitem__(self, index):
        """Returns an example or a sequence of examples.

        It implements the standard Python indexing. It uses the
        :meth:`get_example` method by default, but it may be overridden by the
        implementation to improve the slicing performance.

        """
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            ret = []
            while current < stop and step > 0 or current > stop and step < 0:
                ret.append(self.get_example(current))
                current += step
            return ret
        else:
            return self.get_example(index)

    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError

    def get_example(self, i):
        """Returns the i-th example.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        raise NotImplementedError
