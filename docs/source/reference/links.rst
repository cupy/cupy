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

Bias
~~~~

.. autoclass:: Bias
   :members:

Bilinear
~~~~~~~~
.. autoclass:: Bilinear
   :members:

Convolution2D
~~~~~~~~~~~~~
.. autoclass:: Convolution2D
   :members:

Deconvolution2D
~~~~~~~~~~~~~~~
.. autoclass:: Deconvolution2D
   :members:

EmbedID
~~~~~~~
.. autoclass:: EmbedID
   :members:

GRU
~~~
.. autoclass:: GRU
   :members:

Inception
~~~~~~~~~
.. autoclass:: Inception
   :members:

InceptionBN
~~~~~~~~~~~
.. autoclass:: InceptionBN
   :members:

Linear
~~~~~~
.. autoclass:: Linear
   :members:

LSTM
~~~~
.. autoclass:: LSTM
   :members:

MLPConvolution2D
~~~~~~~~~~~~~~~~
.. autoclass:: MLPConvolution2D
   :members:

Scale
~~~~~
.. autoclass:: Scale
   :members:

StatefulGRU
~~~~~~~~~~~
.. autoclass:: StatefulGRU
   :members:

StatelessLSTM
~~~~~~~~~~~~~
.. autoclass:: StatelessLSTM
   :members:

Activation/loss/normalization functions with parameters
-------------------------------------------------------

BatchNormalization
~~~~~~~~~~~~~~~~~~
.. autoclass:: BatchNormalization
   :members:

BinaryHierarchicalSoftmax
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: BinaryHierarchicalSoftmax
   :members:

PReLU
~~~~~
.. autoclass:: PReLU
   :members:

Maxout
~~~~~~
.. autoclass:: Maxout
   :members:

NegativeSampling
~~~~~~~~~~~~~~~~
.. autoclass:: NegativeSampling
   :members:

Machine learning models
-----------------------

Classifier
~~~~~~~~~~
.. autoclass:: Classifier
   :members:

Deprecated links
----------------

Parameter
~~~~~~~~~
.. autoclass:: Parameter
   :members:
