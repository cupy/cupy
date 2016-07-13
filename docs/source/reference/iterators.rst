.. module:: chainer.iterators

.. _iterators:

Iterator examples
=================

Chainer provides some iterators that implement typical strategies to create minibatches by iterating over datasets.
:class:`SerialIterator` is the simplest one, which extract mini batches in the main thread.
:class:`MultiprocessIterator` is a parallelized version of :class:`SerialIterator`. It maintains worker subprocesses to load the next mini batch in parallel.


SerialIterator
--------------
.. autoclass:: SerialIterator
   :members:

MultiprocessIterator
--------------------
.. autoclass:: MultiprocessIterator
   :members:
