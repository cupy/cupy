.. module:: chainer.dataset

Dataset abstraction
===================

Chainer has a support of common interface of training and validation datasets. The dataset support consists of three components: datasets, iterators, and batch conversion functions.

**Dataset** represents a set of examples. The interface is only determined by combination with iterators you want to use on it. The built-in iterators of Chainer requires the dataset to support ``__getitem__`` and ``__len__`` method. In particular, the ``__getitem__`` method should support indexing by both an integer and a slice. We can easily support slice indexing by inheriting :class:`DatasetMixin`, in which case users only have to implement :meth:`~DatasetMixin.get_example` method for indexing. Some iterators also restrict the type of each example. Basically, datasets are considered as `stateless` objects, so that we do not need to save the dataset as a checkpoint of the training procedure.

**Iterator** iterates over the dataset, and at each iteration, it yields a mini batch of examples as a list. Iterators should support the :class:`Iterator` interface, which includes the standard iterator protocol of Python. Iterators manage where to read next, which means they are `stateful`.

**Batch conversion function** converts the mini batch into arrays to feed to the neural nets. They are also responsible to send each array to an appropriate device. Chainer currently provides :func:`concat_examples` as the only example of batch conversion functions.

These components are all customizable, and designed to have a minimum interface to restrict the types of datasets and ways to handle them. In most cases, though, implementations provided by Chainer itself are enough to cover the usages.

Chainer also has a light system to download, manage, and cache concrete examples of datasets. All datasets managed through the system are saved under `the dataset root directory`, which is determined by the ``CHAINER_DATASET_ROOT`` environment variable, and can also be set by the :func:`set_dataset_root` function.


Dataset representation
~~~~~~~~~~~~~~~~~~~~~~
See :ref:`datasets` for dataset implementations.

.. autoclass:: DatasetMixin
   :members:

Iterator interface
~~~~~~~~~~~~~~~~~~
See :ref:`iterators` for dataset iterator implementations.

.. autoclass:: Iterator
   :members:

Batch conversion function
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: concat_examples
.. autofunction:: to_device

Dataset management
~~~~~~~~~~~~~~~~~~
.. autofunction:: get_dataset_root
.. autofunction:: set_dataset_root
.. autofunction:: cached_download
.. autofunction:: cache_or_load_file
