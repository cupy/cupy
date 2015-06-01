Standard Function implementations
=================================

.. module:: chainer.functions

Chainer provides basic :class:`~chainer.Function` implementations in
:mod:`chainer.functions` package.

Non-parameterized functions are provided as plain Python functions. These can be
directly used in forward computation without explicit handling of
:class:`~chainer.Function` objects. On the other hand, parameterized functions
should be used with explicit handling of :class:`~chainer.Function` objects.

Learnable connections
---------------------
.. autoclass:: Convolution2D
.. autoclass:: EmbedID
.. autoclass:: Linear
.. autoclass:: Parameter

Array manipulation functions
----------------------------
.. autofunction:: concat
.. autofunction:: copy
.. autofunction:: dropout
.. autofunction:: identity
.. autofunction:: reshape

Activation functions
--------------------
.. autofunction:: exp
.. autofunction:: leaky_relu
.. autofunction:: log
.. autofunction:: lstm
.. autoclass:: PReLU
.. autofunction:: relu
.. autofunction:: sigmoid
.. autofunction:: softmax
.. autofunction:: tanh

Pooling functions
-----------------
.. autofunction:: average_pooling_2d
.. autofunction:: max_pooling_2d

Normalization functions
-----------------------
.. autoclass:: BatchNormalization
   :members: __call__
.. autofunction:: local_response_normalization

Loss, evaluation and aggregation
--------------------------------
.. autofunction:: accuracy
.. autofunction:: mean_squared_error
.. autofunction:: softmax_cross_entropy
.. autofunction:: sum

Reusable subnetwork of complex architectures
--------------------------------------------
.. autoclass:: Inception
