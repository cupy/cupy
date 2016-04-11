"""Collection of :class:`~chainer.Function` implementations."""

from chainer.functions.activation import clipped_relu
from chainer.functions.activation import elu
from chainer.functions.activation import leaky_relu
from chainer.functions.activation import log_softmax
from chainer.functions.activation import lstm
from chainer.functions.activation import maxout
from chainer.functions.activation import prelu
from chainer.functions.activation import relu
from chainer.functions.activation import sigmoid
from chainer.functions.activation import slstm
from chainer.functions.activation import softmax
from chainer.functions.activation import softplus
from chainer.functions.activation import tanh
from chainer.functions.array import broadcast
from chainer.functions.array import concat
from chainer.functions.array import copy
from chainer.functions.array import expand_dims
from chainer.functions.array import reshape
from chainer.functions.array import select_item
from chainer.functions.array import split_axis
from chainer.functions.array import swapaxes
from chainer.functions.array import transpose
from chainer.functions.array import where
from chainer.functions.connection import bilinear
from chainer.functions.connection import convolution_2d
from chainer.functions.connection import deconvolution_2d
from chainer.functions.connection import embed_id
from chainer.functions.connection import linear
from chainer.functions.evaluation import accuracy
from chainer.functions.evaluation import binary_accuracy
from chainer.functions.loss import contrastive
from chainer.functions.loss import cross_covariance
from chainer.functions.loss import ctc
from chainer.functions.loss import hinge
from chainer.functions.loss import huber_loss
from chainer.functions.loss import mean_squared_error
from chainer.functions.loss import negative_sampling
from chainer.functions.loss import sigmoid_cross_entropy
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.loss import vae  # NOQA
from chainer.functions.math import basic_math  # NOQA
from chainer.functions.math import batch_l2_norm_squared
from chainer.functions.math import clip
from chainer.functions.math import det
from chainer.functions.math import exponential
from chainer.functions.math import identity
from chainer.functions.math import inv
from chainer.functions.math import matmul
from chainer.functions.math import minmax
from chainer.functions.math import sum
from chainer.functions.math import trigonometric
from chainer.functions.noise import dropout
from chainer.functions.noise import gaussian
from chainer.functions.normalization import batch_normalization
from chainer.functions.normalization import local_response_normalization
from chainer.functions.pooling import average_pooling_2d
from chainer.functions.pooling import max_pooling_2d
from chainer.functions.pooling import spatial_pyramid_pooling_2d
from chainer.functions.pooling import unpooling_2d
from chainer.links.activation import prelu as links_prelu
from chainer.links.connection import bilinear as links_bilinear
from chainer.links.connection import convolution_2d as links_convolution_2d
from chainer.links.connection import embed_id as links_embed_id
from chainer.links.connection import inception
from chainer.links.connection import inceptionbn
from chainer.links.connection import linear as links_linear
from chainer.links.connection import parameter
from chainer.links.loss import hierarchical_softmax
from chainer.links.loss import negative_sampling as links_negative_sampling
from chainer.links.normalization import batch_normalization \
    as links_batch_normalization


ClippedReLU = clipped_relu.ClippedReLU
clipped_relu = clipped_relu.clipped_relu
ConnectionistTemporalClassification = ctc.ConnectionistTemporalClassification
connectionist_temporal_classification \
    = ctc.connectionist_temporal_classification
ELU = elu.ELU
elu = elu.elu
LeakyReLU = leaky_relu.LeakyReLU
leaky_relu = leaky_relu.leaky_relu
LogSoftmax = log_softmax.LogSoftmax
log_softmax = log_softmax.log_softmax
LSTM = lstm.LSTM
lstm = lstm.lstm
maxout = maxout.maxout
prelu = prelu.prelu
ReLU = relu.ReLU
relu = relu.relu
Sigmoid = sigmoid.Sigmoid
sigmoid = sigmoid.sigmoid
SLSTM = slstm.SLSTM
slstm = slstm.slstm
Softmax = softmax.Softmax
softmax = softmax.softmax
Softplus = softplus.Softplus
softplus = softplus.softplus
Tanh = tanh.Tanh
tanh = tanh.tanh

Broadcast = broadcast.Broadcast
BroadcastTo = broadcast.BroadcastTo
broadcast_to = broadcast.broadcast_to
broadcast = broadcast.broadcast
Concat = concat.Concat
concat = concat.concat
Copy = copy.Copy
copy = copy.copy
ExpandDims = expand_dims.ExpandDims
expand_dims = expand_dims.expand_dims
Reshape = reshape.Reshape
reshape = reshape.reshape
SplitAxis = split_axis.SplitAxis
split_axis = split_axis.split_axis
SelectItem = select_item.SelectItem
select_item = select_item.select_item
Swapaxes = swapaxes.Swapaxes
swapaxes = swapaxes.swapaxes
Transpose = transpose.Transpose
transpose = transpose.transpose
Where = where.Where
where = where.where

bilinear = bilinear.bilinear
convolution_2d = convolution_2d.convolution_2d
deconvolution_2d = deconvolution_2d.deconvolution_2d
embed_id = embed_id.embed_id
linear = linear.linear

Accuracy = accuracy.Accuracy
accuracy = accuracy.accuracy
BinaryAccuracy = binary_accuracy.BinaryAccuracy
binary_accuracy = binary_accuracy.binary_accuracy

bernoulli_nll = vae.bernoulli_nll
BinaryHierarchicalSoftmax = hierarchical_softmax.BinaryHierarchicalSoftmax
Contrastive = contrastive.Contrastive
contrastive = contrastive.contrastive
CrossCovariance = cross_covariance.CrossCovariance
cross_covariance = cross_covariance.cross_covariance
gaussian_kl_divergence = vae.gaussian_kl_divergence
gaussian_nll = vae.gaussian_nll
Hinge = hinge.Hinge
hinge = hinge.hinge
MeanSquaredError = mean_squared_error.MeanSquaredError
mean_squared_error = mean_squared_error.mean_squared_error
negative_sampling = negative_sampling.negative_sampling
SigmoidCrossEntropy = sigmoid_cross_entropy.SigmoidCrossEntropy
sigmoid_cross_entropy = sigmoid_cross_entropy.sigmoid_cross_entropy
HuberLoss = huber_loss.HuberLoss
huber_loss = huber_loss.huber_loss
SoftmaxCrossEntropy = softmax_cross_entropy.SoftmaxCrossEntropy
softmax_cross_entropy = softmax_cross_entropy.softmax_cross_entropy

BatchDet = det.BatchDet
batch_det = det.batch_det
BatchInv = inv.BatchInv
batch_inv = inv.batch_inv
BatchL2NormSquared = batch_l2_norm_squared.BatchL2NormSquared
batch_l2_norm_squared = batch_l2_norm_squared.batch_l2_norm_squared
BatchMatMul = matmul.BatchMatMul
batch_matmul = matmul.batch_matmul
Clip = clip.Clip
clip = clip.clip
Cos = trigonometric.Cos
cos = trigonometric.cos
det = det.det
Exp = exponential.Exp
exp = exponential.exp
Identity = identity.Identity
identity = identity.identity
Inv = inv.Inv
inv = inv.inv
Log = exponential.Log
log = exponential.log
MatMul = matmul.MatMul
matmul = matmul.matmul
Max = minmax.Max
max = minmax.max
Min = minmax.Min
min = minmax.min
Sin = trigonometric.Sin
sin = trigonometric.sin
Sum = sum.Sum
sum = sum.sum

Dropout = dropout.Dropout
dropout = dropout.dropout
Gaussian = gaussian.Gaussian
gaussian = gaussian.gaussian

fixed_batch_normalization = batch_normalization.fixed_batch_normalization
batch_normalization = batch_normalization.batch_normalization
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

Unpooling2D = unpooling_2d.Unpooling2D
unpooling_2d = unpooling_2d.unpooling_2d

# Import for backward compatibility
PReLU = links_prelu.PReLU

Bilinear = links_bilinear.Bilinear
Convolution2D = links_convolution_2d.Convolution2D
EmbedID = links_embed_id.EmbedID
Inception = inception.Inception
InceptionBN = inceptionbn.InceptionBN
Linear = links_linear.Linear
Parameter = parameter.Parameter

NegativeSampling = links_negative_sampling.NegativeSampling

BatchNormalization = links_batch_normalization.BatchNormalization
