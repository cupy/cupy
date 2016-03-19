Standard Link implementations
=============================

.. module:: chainer.links

Chainer provides many :class:`~chainer.Link` implementations in the
:mod:`chainer.links` package.

.. note::
   Some of the links are originally defined in the :mod:`chainer.functions`
   namespace. They are still left in the namespace for backward compatibility,
   though it is strongly recommended to use them via the :mod:`chainer.links`
   package.


Learnable connections
---------------------
.. autoclass:: Bilinear
   :members:
.. autoclass:: Convolution2D
   :members:
.. autoclass:: Deconvolution2D
   :members:
.. autoclass:: EmbedID
   :members:
.. autoclass:: GRU
   :members:
.. autoclass:: Inception
   :members:
.. autoclass:: InceptionBN
   :members:
.. autoclass:: Linear
   :members:
.. autoclass:: LSTM
   :members:
.. autoclass:: MLPConvolution2D
   :members:
.. autoclass:: StatefulGRU
   :members:

Activation/loss/normalization functions with parameters
-------------------------------------------------------
.. autoclass:: BatchNormalization
   :members:
.. autoclass:: BinaryHierarchicalSoftmax
   :members:
.. autoclass:: PReLU
   :members:
.. autoclass:: Maxout
   :members:
.. autoclass:: NegativeSampling
   :members:

Machine learning models
-----------------------
.. autoclass:: Classifier
   :members:

Deprecated links
----------------
.. autoclass:: Parameter
   :members:
