Standard Function implementations
=================================

.. module:: chainer.functions

Chainer provides basic :class:`~chainer.Function` implementations in the
:mod:`chainer.functions` package. Most of them are wrapped by plain Python
functions, which users should use.

.. note::
   As of v1.5, the concept of parameterized functions are gone, and they are
   replaced by corresponding :class:`~chainer.Link` implementations. They are
   still put in the :mod:`~chainer.functions` namespace for backward
   compatibility, though it is strongly recommended to use them via the
   :mod:`chainer.links` package.

..
   For contributors that want to update these lists:

   Each list corresponds to the package under chainer.functions. For example,
   the first section "Activation functions" shows functions under the
   chainer.functions.activation subpackage.

   KEEP EACH LIST IN LEXICOGRAPHICAL ORDER.

Activation functions
--------------------
.. autofunction:: clipped_relu
.. autoclass:: GRU
.. autofunction:: leaky_relu
.. autofunction:: lstm
.. autofunction:: prelu
.. autofunction:: relu
.. autofunction:: sigmoid
.. autofunction:: softmax
.. autofunction:: softplus
.. autofunction:: tanh

Array manipulations
-------------------
.. autofunction:: broadcast
.. autofunction:: concat
.. autofunction:: copy
.. autofunction:: reshape
.. autofunction:: select_item
.. autofunction:: split_axis
.. autofunction:: swapaxes
.. autofunction:: transpose
.. autofunction:: where

Neural network connections
--------------------------
.. autofunction:: bilinear
.. autofunction:: convolution_2d
.. autofunction:: embed_id
.. autofunction:: linear

Evaluation functions
--------------------
.. autofunction:: accuracy

Loss functions
--------------
.. autofunction:: connectionist_temporal_classification
.. autofunction:: cross_covariance
.. autofunction:: mean_squared_error
.. autofunction:: negative_sampling
.. autofunction:: sigmoid_cross_entropy
.. autofunction:: softmax_cross_entropy

Loss functions for VAE
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: bernoulli_nll
.. autofunction:: gaussian_kl_divergence
.. autofunction:: gaussian_nll

Mathematical functions
----------------------
.. autofunction:: batch_matmul
.. autofunction:: cos
.. autofunction:: exp
.. autofunction:: identity
.. autofunction:: log
.. autofunction:: matmul
.. autofunction:: max
.. autofunction:: min
.. autofunction:: sin
.. autofunction:: sum

Noise injections
----------------
.. autofunction:: dropout
.. autofunction:: gaussian

Normalization functions
-----------------------
.. autofunction:: batch_normalization
.. autofunction:: fixed_batch_normalization
.. autofunction:: local_response_normalization

Spatial pooling
---------------
.. autofunction:: average_pooling_2d
.. autofunction:: max_pooling_2d
.. autofunction:: spatial_pyramid_pooling_2d
