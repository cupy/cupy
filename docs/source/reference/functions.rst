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

clipped_relu
~~~~~~~~~~~~
.. autofunction:: clipped_relu

crelu
~~~~~
.. autofunction:: crelu

elu
~~~
.. autofunction:: elu

hard_sigmoid
~~~~~~~~~~~~
.. autofunction:: hard_sigmoid

leaky_relu
~~~~~~~~~~
.. autofunction:: leaky_relu

log_softmax
~~~~~~~~~~~
.. autofunction:: log_softmax

lstm
~~~~
.. autofunction:: lstm

maxout
~~~~~~
.. autofunction:: maxout

prelu
~~~~~
.. autofunction:: prelu

relu
~~~~
.. autofunction:: relu

sigmoid
~~~~~~~
.. autofunction:: sigmoid

slstm
~~~~~
.. autofunction:: slstm

softmax
~~~~~~~
.. autofunction:: softmax

softplus
~~~~~~~~
.. autofunction:: softplus

tanh
~~~~
.. autofunction:: tanh


Array manipulations
-------------------

broadcast
~~~~~~~~~
.. autofunction:: broadcast

broadcast_to
~~~~~~~~~~~~
.. autofunction:: broadcast_to

cast
~~~~
.. autofunction:: cast

concat
~~~~~~
.. autofunction:: concat

copy
~~~~
.. autofunction:: copy

dstack
~~~~~~
.. autofunction:: dstack

expand_dims
~~~~~~~~~~~
.. autofunction:: expand_dims

flatten
~~~~~~~
.. autofunction:: flatten

get_item
~~~~~~~~
.. autofunction:: get_item

hstack
~~~~~~
.. autofunction:: hstack

permutate
~~~~~~~~~
.. autofunction:: permutate

reshape
~~~~~~~
.. autofunction:: reshape

rollaxis
~~~~~~~~
.. autofunction:: rollaxis

select_item
~~~~~~~~~~~
.. autofunction:: select_item

separate
~~~~~~~~
.. autofunction:: separate

split_axis
~~~~~~~~~~
.. autofunction:: split_axis

squeeze
~~~~~~~
.. autofunction:: squeeze

stack
~~~~~
.. autofunction:: stack

swapaxes
~~~~~~~~
.. autofunction:: swapaxes

tile
~~~~
.. autofunction:: tile

transpose
~~~~~~~~~
.. autofunction:: transpose

transpose_sequence
~~~~~~~~~~~~~~~~~~
.. autofunction:: transpose_sequence

vstack
~~~~~~
.. autofunction:: vstack

where
~~~~~
.. autofunction:: where


Neural network connections
--------------------------

bilinear
~~~~~~~~
.. autofunction:: bilinear

convolution_2d
~~~~~~~~~~~~~~
.. autofunction:: convolution_2d

convolution_nd
~~~~~~~~~~~~~~
.. autofunction:: convolution_nd

deconvolution_2d
~~~~~~~~~~~~~~~~
.. autofunction:: deconvolution_2d

deconvolution_nd
~~~~~~~~~~~~~~~~
.. autofunction:: deconvolution_nd

dilated_convolution_2d
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: dilated_convolution_2d

embed_id
~~~~~~~~
.. autofunction:: embed_id

linear
~~~~~~
.. autofunction:: linear

n_step_lstm
~~~~~~~~~~~
.. autofunction:: n_step_lstm


Evaluation functions
--------------------

accuracy
~~~~~~~~
.. autofunction:: accuracy

binary_accuracy
~~~~~~~~~~~~~~~
.. autofunction:: binary_accuracy

classification_summary
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: classification_summary

r2_score
~~~~~~~~
.. autofunction:: r2_score


Loss functions
--------------

bernoulli_nll
~~~~~~~~~~~~~
.. autofunction:: bernoulli_nll

black_out
~~~~~~~~~
.. autofunction:: black_out

connectionist_temporal_classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: connectionist_temporal_classification

contrastive
~~~~~~~~~~~
.. autofunction:: contrastive

crf1d
~~~~~
.. autofunction:: crf1d
.. autofunction:: argmax_crf1d

cross_covariance
~~~~~~~~~~~~~~~~
.. autofunction:: cross_covariance

gaussian_kl_divergence
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: gaussian_kl_divergence

gaussian_nll
~~~~~~~~~~~~
.. autofunction:: gaussian_nll

hinge
~~~~~
.. autofunction:: hinge

huber_loss
~~~~~~~~~~
.. autofunction:: huber_loss

mean_absolute_error
~~~~~~~~~~~~~~~~~~~
.. autofunction:: mean_absolute_error

mean_squared_error
~~~~~~~~~~~~~~~~~~
.. autofunction:: mean_squared_error

negative_sampling
~~~~~~~~~~~~~~~~~
.. autofunction:: negative_sampling

sigmoid_cross_entropy
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: sigmoid_cross_entropy

softmax_cross_entropy
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: softmax_cross_entropy

triplet
~~~~~~~
.. autofunction:: triplet


Mathematical functions
----------------------

arccos
~~~~~~
.. autofunction:: arccos

arcsin
~~~~~~
.. autofunction:: arcsin

arctan
~~~~~~
.. autofunction:: arctan

argmax
~~~~~~
.. autofunction:: argmax

argmin
~~~~~~
.. autofunction:: argmin

batch_inv
~~~~~~~~~
.. autofunction:: batch_inv

batch_l2_norm_squared
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: batch_l2_norm_squared

batch_matmul
~~~~~~~~~~~~
.. autofunction:: batch_matmul

bias
~~~~
.. autofunction:: bias

ceil
~~~~
.. autofunction:: ceil

clip
~~~~
.. autofunction:: clip

cos
~~~
.. autofunction:: cos

cosh
~~~~
.. autofunction:: cosh

exp
~~~
.. autofunction:: exp

fmod
~~~~
.. autofunction:: fmod

floor
~~~~~
.. autofunction:: floor

identity
~~~~~~~~
.. autofunction:: identity

inv
~~~
.. autofunction:: inv

linear_interpolate
~~~~~~~~~~~~~~~~~~
.. autofunction:: linear_interpolate

log
~~~
.. autofunction:: log

log10
~~~~~
.. autofunction:: log10

log1p
~~~~~
.. autofunction:: log1p

log2
~~~~
.. autofunction:: log2

logsumexp
~~~~~~~~~
.. autofunction:: logsumexp

matmul
~~~~~~
.. autofunction:: matmul

max
~~~
.. autofunction:: max

maximum
~~~~~~~
.. autofunction:: maximum

min
~~~
.. autofunction:: min

minimum
~~~~~~~
.. autofunction:: minimum

rsqrt
~~~~~
.. autofunction:: rsqrt

scale
~~~~~
.. autofunction:: scale

sin
~~~
.. autofunction:: sin

sinh
~~~~
.. autofunction:: sinh

sqrt
~~~~
.. autofunction:: sqrt

square
~~~~~~
.. autofunction:: square

squared_difference
~~~~~~~~~~~~~~~~~~
.. autofunction:: squared_difference

sum
~~~
.. autofunction:: sum

tanh
~~~~
Hyperbolic tangent function is described in "Activation functions" section.

.. seealso:: :func:`~chainer.functions.tanh`

tan
~~~
.. autofunction:: tan


Noise injections
----------------

dropout
~~~~~~~
.. autofunction:: dropout

gaussian
~~~~~~~~
.. autofunction:: gaussian


Normalization functions
-----------------------

batch_normalization
~~~~~~~~~~~~~~~~~~~
.. autofunction:: batch_normalization

fixed_batch_normalization
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: fixed_batch_normalization

local_response_normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: local_response_normalization

normalize
~~~~~~~~~
.. autofunction:: normalize


Spatial pooling
---------------

average_pooling_2d
~~~~~~~~~~~~~~~~~~
.. autofunction:: average_pooling_2d

average_pooling_nd
~~~~~~~~~~~~~~~~~~
.. autofunction:: average_pooling_nd

max_pooling_2d
~~~~~~~~~~~~~~
.. autofunction:: max_pooling_2d

max_pooling_nd
~~~~~~~~~~~~~~
.. autofunction:: max_pooling_nd

roi_pooling_2d
~~~~~~~~~~~~~~
.. autofunction:: roi_pooling_2d

spatial_pyramid_pooling_2d
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: spatial_pyramid_pooling_2d

unpooling_2d
~~~~~~~~~~~~
.. autofunction:: unpooling_2d

upsampling_2d
~~~~~~~~~~~~~
.. autofunction:: upsampling_2d


Utility functions
-----------------

forget
~~~~~~
.. autofunction:: forget
