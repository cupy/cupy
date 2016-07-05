.. module:: chainer.iterators

.. _iterators:

Iterator examples
=================

Chainer provides some iterators that implement typical strategies to create minibatches by iterating over datasets.
:class:`SequentialIterator` is the simplest one, which sequentially visit each example in a dataset.
:class:`ShuffledIterator` shuffles the examples at the beginning of each epoch (i.e., one sweep over the whole dataset).
:class:`MultiprocessIterator` is a parallelized version of :class:`ShuffledIterator`. It maintains worker subprocesses to load the next mini batch in parallel.


SequentialIterator
------------------
.. autoclass:: SequentialIterator
   :members:

ShuffledIterator
----------------
.. autoclass:: ShuffledIterator
   :members:

MultiprocessIterator
--------------------
.. autoclass:: MultiprocessIterator
   :members:
