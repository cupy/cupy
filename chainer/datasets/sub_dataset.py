import numpy
import six

from chainer.dataset import dataset_mixin


class SubDataset(dataset_mixin.DatasetMixin):

    """Subset of a base dataset.

    SubDataset defines a subset of a given base dataset. The subset is defined
    as an interval of indexes, optionally with a given permutation.

    If ``order`` is given, then the ``i``-th example of this dataset is the
    ``order[start + i]``-th example of the base dataset, where ``i`` is a
    non-negative integer. If ``order`` is not given, then the ``i``-th example
    of this dataset is the ``start + i``-th example of the base dataset.
    Negative indexing is also allowed: in this case, the term ``start + i`` is
    replaced by ``finish + i``.

    SubDataset is often used to split a dataset into training and validation
    subsets. The training set is used for training, while the validation set is
    used to track the generalization performance, i.e. how the learned model
    works well on unseen data. We can tune hyperparameters (e.g. number of
    hidden units, weight initializers, learning rate, etc.) by comparing the
    validation performance. Note that we often use another set called test set
    to measure the quality of the tuned hyperparameter, which can be made by
    nesting multiple SubDatasets.

    There are two ways to make training-validation splits. One is a single
    split, where the dataset is split just into two subsets. It can be done by
    :func:`split_dataset` or :func:`split_dataset_random`. The other one is a
    :math:`k`-fold cross validation, in which the dataset is divided into
    :math:`k` subsets, and :math:`k` different splits are generated using each
    of the :math:`k` subsets as a validation set and the rest as a training
    set. It can be done by :func:`get_cross_validation_datasets`.

    Args:
        dataset: Base dataset.
        start (int): The first index in the interval.
        finish (int): The next-to-the-last index in the interval.
        order (sequence of ints): Permutation of indexes in the base dataset.
            If this is ``None``, then the ascending order of indexes is used.

    """

    def __init__(self, dataset, start, finish, order=None):
        if start < 0 or finish > len(dataset):
            raise ValueError('subset overruns the base dataset.')
        self._dataset = dataset
        self._start = start
        self._finish = finish
        self._size = finish - start
        if order is not None and len(order) != len(dataset):
            msg = ('order option must have the same length as the base '
                   'dataset: len(order) = {} while len(dataset) = {}'.format(
                       len(order), len(dataset)))
            raise ValueError(msg)
        self._order = order

    def __len__(self):
        return self._size

    def get_example(self, i):
        if i >= 0:
            if i >= self._size:
                raise IndexError('dataset index out of range')
            index = self._start + i
        else:
            if i < -self._size:
                raise IndexError('dataset index out of range')
            index = self._finish + i

        if self._order is not None:
            index = self._order[index]
        return self._dataset[index]


def split_dataset(dataset, split_at, order=None):
    """Splits a dataset into two subsets.

    This function creates two instances of :class:`SubDataset`. These instances
    do not share any examples, and they together cover all examples of the
    original dataset.

    Args:
        dataset: Dataset to split.
        split_at (int): Position at which the base dataset is split.
        order (sequence of ints): Permutation of indexes in the base dataset.
            See the document of :class:`SubDataset` for details.

    Returns:
        tuple: Two :class:`SubDataset` objects. The first subset represents the
            examples of indexes ``order[:split_at]`` while the second subset
            represents the examples of indexes ``order[split_at:]``.

    """
    n_examples = len(dataset)
    if split_at < 0:
        raise ValueError('split_at must be non-negative')
    if split_at >= n_examples:
        raise ValueError('split_at exceeds the dataset size')
    subset1 = SubDataset(dataset, 0, split_at, order)
    subset2 = SubDataset(dataset, split_at, n_examples, order)
    return subset1, subset2


def split_dataset_random(dataset, first_size, seed=None):
    """Splits a dataset into two subsets randomly.

    This function creates two instances of :class:`SubDataset`. These instances
    do not share any examples, and they together cover all examples of the
    original dataset. The split is automatically done randomly.

    Args:
        dataset: Dataset to split.
        first_size (int): Size of the first subset.
        seed (int): Seed the generator used for the permutation of indexes.
            If an integer beging convertible to 32 bit unsigned integers is
            specified, it is guaranteed that each sample
            in the given dataset always belongs to a specific subset.
            If ``None``, the permutation is changed randomly.

    Returns:
        tuple: Two :class:`SubDataset` objects. The first subset contains
            ``first_size`` examples randomly chosen from the dataset without
            replacement, and the second subset contains the rest of the
            dataset.

    """
    order = numpy.random.RandomState(seed).permutation(len(dataset))
    return split_dataset(dataset, first_size, order)


def get_cross_validation_datasets(dataset, n_fold, order=None):
    """Creates a set of training/test splits for cross validation.

    This function generates ``n_fold`` splits of the given dataset. The first
    part of each split corresponds to the training dataset, while the second
    part to the test dataset. No pairs of test datasets share any examples, and
    all test datasets together cover the whole base dataset. Each test dataset
    contains almost same number of examples (the numbers may differ up to 1).

    Args:
        dataset: Dataset to split.
        n_fold (int): Number of splits for cross validation.
        order (sequence of ints): Order of indexes with which each split is
            determined. If it is ``None``, then no permutation is used.

    Returns:
        list of tuples: List of dataset splits.

    """
    if order is None:
        order = numpy.arange(len(dataset))
    else:
        order = numpy.array(order)  # copy

    whole_size = len(dataset)
    borders = [whole_size * i // n_fold for i in six.moves.range(n_fold + 1)]
    test_sizes = [borders[i + 1] - borders[i] for i in six.moves.range(n_fold)]

    splits = []
    for test_size in reversed(test_sizes):
        size = whole_size - test_size
        splits.append(split_dataset(dataset, size, order))
        new_order = numpy.empty_like(order)
        new_order[:test_size] = order[-test_size:]
        new_order[test_size:] = order[:-test_size]
        order = new_order

    return splits


def get_cross_validation_datasets_random(dataset, n_fold, seed=None):
    """Creates a set of training/test splits for cross validation randomly.

    This function acts almost same as :func:`get_cross_validation_dataset`,
    except automatically generating random permutation.

    Args:
        dataset: Dataset to split.
        n_fold (int): Number of splits for cross validation.
        seed (int): Seed the generator used for the permutation of indexes.
            If an integer beging convertible to 32 bit unsigned integers is
            specified, it is guaranteed that each sample
            in the given dataset always belongs to a specific subset.
            If ``None``, the permutation is changed randomly.

    Returns:
        list of tuples: List of dataset splits.

    """
    order = numpy.random.RandomState(seed).permutation(len(dataset))
    return get_cross_validation_datasets(dataset, n_fold, order)
