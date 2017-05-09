.. module:: chainer.datasets

.. _datasets:

Dataset examples
================

The most basic :mod:`~chainer.dataset` implementation is an array.
Both NumPy and CuPy arrays can be used directly as datasets.

In many cases, though, the simple arrays are not enough to write the training procedure.
In order to cover most of such cases, Chainer provides many built-in implementations of datasets.

These built-in datasets are divided into two groups.
One is a group of general datasets.
Most of them are wrapper of other datasets to introduce some structures (e.g., tuple or dict) to each data point.
The other one is a group of concrete, popular datasets.
These concrete examples use the downloading utilities in the :mod:`chainer.dataset` module to cache downloaded and converted datasets.


General datasets
----------------

General datasets are further divided into three types.

The first one is :class:`DictDataset` and :class:`TupleDataset`, both of which combine other datasets and introduce some structures on them.

The second one is :class:`SubDataset`, which represents a subset of an existing dataset. It can be used to separate a dataset for hold-out validation or cross validation. Convenient functions to make random splits are also provided.

The third one is :class:`TransformDataset`, which wraps around a dataset by applying a function to data indexed from the underlying dataset.
It can be used to modify behavior of a dataset that is already prepared.

The last one is a group of domain-specific datasets. Currently, :class:`ImageDataset` and :class:`LabeledImageDataset` are provided for datasets of images.


DictDataset
~~~~~~~~~~~
.. autoclass:: DictDataset
   :members:

TupleDataset
~~~~~~~~~~~~
.. autoclass:: TupleDataset
   :members:

SubDataset
~~~~~~~~~~
.. autoclass:: SubDataset
   :members:

.. autofunction:: split_dataset
.. autofunction:: split_dataset_random
.. autofunction:: get_cross_validation_datasets
.. autofunction:: get_cross_validation_datasets_random

TransformDataset
~~~~~~~~~~~~~~~~
.. autoclass:: TransformDataset
   :members:

ImageDataset
~~~~~~~~~~~~
.. autoclass:: ImageDataset
   :members:

LabeledImageDataset
~~~~~~~~~~~~~~~~~~~
.. autoclass:: LabeledImageDataset
   :members:


Concrete datasets
-----------------

MNIST
~~~~~
.. autofunction:: get_mnist

CIFAR10/100
~~~~~~~~~~~
.. autofunction:: get_cifar10
.. autofunction:: get_cifar100

Penn Tree Bank
~~~~~~~~~~~~~~
.. autofunction:: get_ptb_words
.. autofunction:: get_ptb_words_vocabulary
