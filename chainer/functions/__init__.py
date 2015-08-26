"""Collection of :class:`~chainer.Function` implementations."""

from chainer.functions.activation import clipped_relu
from chainer.functions.activation import leaky_relu
from chainer.functions.activation import lstm
from chainer.functions.activation import prelu
from chainer.functions.activation import relu
from chainer.functions.activation import sigmoid
from chainer.functions.activation import softmax
from chainer.functions.activation import softplus
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.array import copy
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.functions.connection import bilinear
from chainer.functions.connection import convolution_2d
from chainer.functions.connection import embed_id
from chainer.functions.connection import inception
from chainer.functions.connection import inceptionbn
from chainer.functions.connection import linear
from chainer.functions.connection import parameter
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import cross_covariance
from chainer.functions.loss import hierarchical_softmax
from chainer.functions.loss import mean_squared_error
from chainer.functions.loss import negative_sampling
from chainer.functions.loss import sigmoid_cross_entropy
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.loss import vae  # NOQA
from chainer.functions.math import basic_math  # NOQA
from chainer.functions.math import exponential
from chainer.functions.math import identity
from chainer.functions.math import matmul
from chainer.functions.math import sum
from chainer.functions.math import trigonometric
from chainer.functions.noise import dropout
from chainer.functions.noise import gaussian
from chainer.functions.normalization import batch_normalization
from chainer.functions.normalization import local_response_normalization
from chainer.functions.pooling import average_pooling_2d
from chainer.functions.pooling import max_pooling_2d
from chainer.functions.pooling import spatial_pyramid_pooling_2d


ClippedReLU = clipped_relu.ClippedReLU
clipped_relu = clipped_relu.clipped_relu
LeakyReLU = leaky_relu.LeakyReLU
leaky_relu = leaky_relu.leaky_relu
LSTM = lstm.LSTM
lstm = lstm.lstm
PReLU = prelu.PReLU
ReLU = relu.ReLU
relu = relu.relu
Sigmoid = sigmoid.Sigmoid
sigmoid = sigmoid.sigmoid
Softmax = softmax.Softmax
softmax = softmax.softmax
Softplus = softplus.Softplus
softplus = softplus.softplus
Tanh = tanh.Tanh
tanh = tanh.tanh

Concat = concat.Concat
concat = concat.concat
Copy = copy.Copy
copy = copy.copy
Reshape = reshape.Reshape
reshape = reshape.reshape
SplitAxis = split_axis.SplitAxis
split_axis = split_axis.split_axis

Bilinear = bilinear.Bilinear
Convolution2D = convolution_2d.Convolution2D
convolution_2d = convolution_2d.convolution_2d
EmbedID = embed_id.EmbedID
Inception = inception.Inception
InceptionBN = inceptionbn.InceptionBN
Linear = linear.Linear
linear = linear.linear
Parameter = parameter.Parameter

Accuracy = accuracy.Accuracy
accuracy = accuracy.accuracy

bernoulli_nll = vae.bernoulli_nll
BinaryHierarchicalSoftmax = hierarchical_softmax.BinaryHierarchicalSoftmax
CrossCovariance = cross_covariance.CrossCovariance
cross_covariance = cross_covariance.cross_covariance
gaussian_kl_divergence = vae.gaussian_kl_divergence
gaussian_nll = vae.gaussian_nll
MeanSquaredError = mean_squared_error.MeanSquaredError
mean_squared_error = mean_squared_error.mean_squared_error
NegativeSampling = negative_sampling.NegativeSampling
SigmoidCrossEntropy = sigmoid_cross_entropy.SigmoidCrossEntropy
sigmoid_cross_entropy = sigmoid_cross_entropy.sigmoid_cross_entropy
SoftmaxCrossEntropy = softmax_cross_entropy.SoftmaxCrossEntropy
softmax_cross_entropy = softmax_cross_entropy.softmax_cross_entropy

BatchMatMul = matmul.BatchMatMul
batch_matmul = matmul.batch_matmul
Cos = trigonometric.Cos
cos = trigonometric.cos
Exp = exponential.Exp
exp = exponential.exp
Identity = identity.Identity
identity = identity.identity
Log = exponential.Log
log = exponential.log
MatMul = matmul.MatMul
matmul = matmul.matmul
Sin = trigonometric.Sin
sin = trigonometric.sin
Sum = sum.Sum
sum = sum.sum

Dropout = dropout.Dropout
dropout = dropout.dropout
Gaussian = gaussian.Gaussian
gaussian = gaussian.gaussian

BatchNormalization = batch_normalization.BatchNormalization
LocalResponseNormalization = \
    local_response_normalization.LocalResponseNormalization
local_response_normalization = \
    local_response_normalization.local_response_normalization

AveragePooling2D = average_pooling_2d.AveragePooling2D
average_pooling_2d = average_pooling_2d.average_pooling_2d
MaxPooling2D = max_pooling_2d.MaxPooling2D
max_pooling_2d = max_pooling_2d.max_pooling_2d
SpatialPyramidPooling2D = spatial_pyramid_pooling_2d.SpatialPyramidPooling2D
spatial_pyramid_pooling_2d = \
    spatial_pyramid_pooling_2d.spatial_pyramid_pooling_2d
