Standard Function implementations
=================================

.. module:: chainer.functions

Chainer provides basic :class:`~chainer.Function` implementations in the
:mod:`chainer.functions` package.

Non-parameterized functions are provided as plain Python functions. These can be
used directly in forward computation without explicit handling of
:class:`~chainer.Function` objects. On the other hand, parameterized functions
should be used with explicit handling of :class:`~chainer.Function` objects.

Learnable connections
---------------------
.. autoclass:: BinaryHierarchicalSoftmax
.. autoclass:: Convolution2D
.. autoclass:: EmbedID
.. autoclass:: Linear
.. autoclass:: NegativeSampling
.. autoclass:: Parameter

Array commputation functions
----------------------------
.. autofunction:: convolution_2d
.. autofunction:: linear

Array manipulation functions
----------------------------
.. autofunction:: concat
.. autofunction:: copy
.. autofunction:: dropout
.. autofunction:: identity
.. autofunction:: reshape
.. autofunction:: split_axis

Array computations
------------------
.. autofunction:: matmul
.. autofunction:: batch_matmul

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
.. autofunction:: sigmoid_cross_entropy
.. autofunction:: softmax_cross_entropy
.. autofunction:: sum
.. autofunction:: cross_covariance

Reusable subnetwork of complex architectures
--------------------------------------------
.. autoclass:: Inception
.. autoclass:: InceptionBN
