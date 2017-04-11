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

ConvolutionND
~~~~~~~~~~~~~
.. autoclass:: ConvolutionND
   :members:

Deconvolution2D
~~~~~~~~~~~~~~~
.. autoclass:: Deconvolution2D
   :members:

DeconvolutionND
~~~~~~~~~~~~~~~
.. autoclass:: DeconvolutionND

DilatedConvolution2D
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: DilatedConvolution2D
   :members:

EmbedID
~~~~~~~
.. autoclass:: EmbedID
   :members:

GRU
~~~
.. autoclass:: GRU
   :members:

Highway
~~~~~~~
.. autoclass:: Highway
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

NStepLSTM
~~~~~~~~~

.. autoclass:: NStepLSTM
   :members:

Scale
~~~~~
.. autoclass:: Scale
   :members:

StatefulGRU
~~~~~~~~~~~
.. autoclass:: StatefulGRU
   :members:

StatefulPeepholeLSTM
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: StatefulPeepholeLSTM
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

LayerNormalization
~~~~~~~~~~~~~~~~~~
.. autoclass:: LayerNormalization
   :members:

BinaryHierarchicalSoftmax
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: BinaryHierarchicalSoftmax
   :members:

BlackOut
~~~~~~~~
.. autoclass:: BlackOut

CRF1d
~~~~~
.. autoclass:: CRF1d
   :members:

SimplifiedDropconnect
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SimplifiedDropconnect
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

Pre-trained models
------------------

Pre-trained models are mainly used to achieve a good performance with a small
dataset, or extract a semantic feature vector. Although ``CaffeFunction``
automatically loads a pre-trained model released as a caffemodel,
the following link models provide an interface for automatically converting
caffemodels, and easily extracting semantic feature vectors.

For example, to extract the feature vectors with ``VGG16Layers``, which is
a common pre-trained model in the field of image recognition,
users need to write the following few lines::

    from chainer.links import VGG16Layers
    from PIL import Image

    model = VGG16Layers()
    img = Image.open("path/to/image.jpg")
    feature = model.extract([img], layers=["fc7"])["fc7"]

where ``fc7`` denotes a layer before the last fully-connected layer.
Unlike the usual links, these classes automatically load all the
parameters from the pre-trained models during initialization.

VGG16Layers
~~~~~~~~~~~
.. autoclass:: VGG16Layers
   :members:

.. autofunction:: chainer.links.model.vision.vgg.prepare

Residual Networks
~~~~~~~~~~~~~~~~~
.. autoclass:: chainer.links.model.vision.resnet.ResNetLayers
   :members:

.. autoclass:: ResNet50Layers
   :members:

.. autoclass:: ResNet101Layers
   :members:

.. autoclass:: ResNet152Layers
   :members:

.. autofunction:: chainer.links.model.vision.resnet.prepare

Deprecated links
----------------

Parameter
~~~~~~~~~
.. autoclass:: Parameter
   :members:
