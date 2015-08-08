"""Collection of :class:`~chainer.Function` implementations."""

from chainer.functions import accuracy
from chainer.functions import basic_math
from chainer.functions import batch_normalization
from chainer.functions import concat
from chainer.functions import convolution_2d
from chainer.functions import copy
from chainer.functions import cross_covariance
from chainer.functions import dropout
from chainer.functions import embed_id
from chainer.functions import gaussian
from chainer.functions import hierarchical_softmax
from chainer.functions import identity
from chainer.functions import inception
from chainer.functions import inceptionbn
from chainer.functions import leaky_relu
from chainer.functions import linear
from chainer.functions import local_response_normalization
from chainer.functions import lstm
from chainer.functions import matmul
from chainer.functions import mean_squared_error
from chainer.functions import negative_sampling
from chainer.functions import nonparameterized_convolution_2d
from chainer.functions import nonparameterized_linear
from chainer.functions import parameter
from chainer.functions import pooling_2d
from chainer.functions import prelu
from chainer.functions import relu
from chainer.functions import reshape
from chainer.functions import sigmoid
from chainer.functions import sigmoid_cross_entropy
from chainer.functions import softmax
from chainer.functions import softmax_cross_entropy
from chainer.functions import softplus
from chainer.functions import split_axis
from chainer.functions import sum as sum_
from chainer.functions import tanh

NonparameterizedConvolution2D = \
    nonparameterized_convolution_2d.NonparameterizedConvolution2D
NonparameterizedLinear = \
    nonparameterized_linear.NonparameterizedLinear
Concat = concat.Concat
Copy = copy.Copy
Cos = basic_math.Cos
CrossCovariance = cross_covariance.CrossCovariance
Dropout = dropout.Dropout
Gaussian = gaussian.Gaussian
Identity = identity.Identity
Reshape = reshape.Reshape
BatchMatMul = matmul.BatchMatMul
SplitAxis = split_axis.SplitAxis
Exp = basic_math.Exp
Log = basic_math.Log
LeakyReLU = leaky_relu.LeakyReLU
LSTM = lstm.LSTM
MatMul = matmul.MatMul
ReLU = relu.ReLU
Sigmoid = sigmoid.Sigmoid
Sin = basic_math.Sin
Softmax = softmax.Softmax
Softplus = softplus.Softplus
Tanh = tanh.Tanh
AveragePooling2D = pooling_2d.AveragePooling2D
MaxPooling2D = pooling_2d.MaxPooling2D
SpatialPyramidPooling2D = pooling_2d.SpatialPyramidPooling2D
Pooling2D = pooling_2d.Pooling2D
LocalResponseNormalization = \
    local_response_normalization.LocalResponseNormalization
Accuracy = accuracy.Accuracy
MeanSquaredError = mean_squared_error.MeanSquaredError
SigmoidCrossEntropy = sigmoid_cross_entropy.SigmoidCrossEntropy
SoftmaxCrossEntropy = softmax_cross_entropy.SoftmaxCrossEntropy
Sum = sum_.Sum
Inception = inception.Inception
InceptionBN = inceptionbn.InceptionBN

BatchNormalization = batch_normalization.BatchNormalization
Convolution2D = convolution_2d.Convolution2D
EmbedID = embed_id.EmbedID
BinaryHierarchicalSoftmax = hierarchical_softmax.BinaryHierarchicalSoftmax
create_huffman_tree = hierarchical_softmax.create_huffman_tree
Linear = linear.Linear
NegativeSampling = negative_sampling.NegativeSampling
Parameter = parameter.Parameter
PReLU = prelu.PReLU

convolution_2d = nonparameterized_convolution_2d.convolution_2d
linear = nonparameterized_linear.linear

concat = concat.concat
copy = copy.copy
dropout = dropout.dropout
gaussian = gaussian.gaussian
identity = identity.identity
reshape = reshape.reshape
split_axis = split_axis.split_axis

absolute = basic_math.absolute
batch_matmul = matmul.batch_matmul
cos = basic_math.cos
exp = basic_math.exp
log = basic_math.log
leaky_relu = leaky_relu.leaky_relu
lstm = lstm.lstm
matmul = matmul.matmul
relu = relu.relu
sigmoid = sigmoid.sigmoid
sin = basic_math.sin
softmax = softmax.softmax
softplus = softplus.softplus
tanh = tanh.tanh

average_pooling_2d = pooling_2d.average_pooling_2d
max_pooling_2d = pooling_2d.max_pooling_2d
spatial_pyramid_pooling_2d = pooling_2d.spatial_pyramid_pooling_2d
local_response_normalization = \
    local_response_normalization.local_response_normalization

accuracy = accuracy.accuracy
cross_covariance = cross_covariance.cross_covariance
mean_squared_error = mean_squared_error.mean_squared_error
sigmoid_cross_entropy = sigmoid_cross_entropy.sigmoid_cross_entropy
softmax_cross_entropy = softmax_cross_entropy.softmax_cross_entropy
sum = sum_.sum
