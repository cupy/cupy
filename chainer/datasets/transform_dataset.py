from chainer.dataset import dataset_mixin


class TransformDataset(dataset_mixin.DatasetMixin):

    """Dataset that indexes the base dataset and transforms the data.

    This dataset wraps the base dataset by modifying the behavior of the base
    dataset's :meth:`__getitem__`. Arrays returned by :meth:`__getitem__` of
    the base dataset with integer as an argument are transformed by the given
    function :obj:`transform`.
    Also, :meth:`__len__` returns the integer returned by the base dataset's
    :meth:`__len__`.

    The function :obj:`transform` takes, as an argument, :obj:`in_data`, which
    is the output of the base dataset's :meth:`__getitem__`, and returns
    the transformed arrays as output. Please see the following example.

    >>> from chainer.datasets import get_mnist
    >>> from chainer.datasets import TransformDataset
    >>> dataset, _ = get_mnist()
    >>> def transform(in_data):
    ...     img, label = in_data
    ...     img -= 0.5  # scale to [-0.5, -0.5]
    ...     return img, label
    >>> dataset = TransformDataset(dataset, transform)

    Args:
        dataset: The underlying dataset. The index of this dataset corresponds
            to the index of the base dataset. This object needs to support
            functions :meth:`__getitem__` and :meth:`__len__` as described
            above.
        transform (callable): A function that is called to transform values
            returned by the underlying dataset's :meth:`__getitem__`.

    """

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        in_data = self._dataset[i]
        return self._transform(in_data)
