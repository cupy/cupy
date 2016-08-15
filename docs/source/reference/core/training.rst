.. module:: chainer.training

Training loop abstraction
=========================

Chainer provides a standard implementation of the training loops under the :mod:`chainer.training` module. It is built on top of many other core features of Chainer, including Variable and Function, Link/Chain/ChainList, Optimizer, Dataset, and Reporter/Summary. Compared to the training loop abstraction of other machine learning tool kits, Chainer's training framework aims at maximal flexibility, while keeps the simplicity for the typical usages. Most components are pluggable, and users can overwrite the definition.

The core of the training loop abstraction is :class:`Trainer`, which implements the training loop itself. The training loop consists of two parts: one is :class:`Updater`, which actually updates the parameters to train, and the other is :class:`Extension` for arbitrary functionalities other than the parameter update.

Updater and some extensions use :mod:`~chainer.dataset` and :class:`~chainer.Iterator` to scan the datasets and load mini batches. The trainer also uses :class:`~chainer.Reporter` to collect the observed values, and some extensions use :class:`~chainer.DictSummary` to accumulate them and computes the statistics.

You can find many examples for the usage of this training utilities from the official examples. You can also search the extension implementations from :ref:`extensions`.


Trainer
-------
.. autoclass:: Trainer
   :members:


Updater
-------
.. autoclass:: Updater
   :members:

.. autoclass:: StandardUpdater
   :members:

.. autoclass:: ParallelUpdater
   :members:


Extension
---------
.. autoclass:: Extension
   :members:

.. autofunction:: make_extension


Trigger
-------
Trigger is a callable object to decide when to process some specific event within the training loop. It takes a Trainer object as the argument, and returns True if some event should be fired.

It is mainly used to determine when to call an extension. It is also used to determine when to quit the training loop.

.. autoclass:: IntervalTrigger
   :members:

.. autofunction:: get_trigger
