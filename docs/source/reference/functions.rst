Standard Function implementations
=================================

.. currentmodule:: chainer.functions

Parameterized function classes
------------------------------
.. autoclass:: BatchNormalization
.. autoclass:: Convolution2D
.. autoclass:: EmbedID
.. autoclass:: Linear
.. autoclass:: Parameter
.. autoclass:: PReLU

Array manipulation functions
----------------------------
.. autofunction:: concat
.. autofunction:: copy
.. autofunction:: dropout
.. autofunction:: identity

Activation functions
--------------------
.. autofunction:: leaky_relu
.. autofunction:: lstm
.. autofunction:: relu
.. autofunction:: sigmoid
.. autofunction:: softmax
.. autofunction:: tanh

Pooling functions
-----------------
.. autofunction:: average_pooling_2d
.. autofunction:: max_pooling_2d

Loss, evaluation and aggregation
--------------------------------
.. autofunction:: accuracy
.. autofunction:: mean_squared_error
.. autofunction:: softmax_cross_entropy
.. autofunction:: sum
