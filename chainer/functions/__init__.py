"""Collection of :class:`~chainer.Function` implementations."""

from chainer.functions.activation import clipped_relu
from chainer.functions.activation import crelu
from chainer.functions.activation import elu
from chainer.functions.activation import hard_sigmoid
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
from chainer.functions.array import cast
from chainer.functions.array import concat
from chainer.functions.array import copy
from chainer.functions.array import dstack
from chainer.functions.array import expand_dims
from chainer.functions.array import flatten
from chainer.functions.array import get_item
from chainer.functions.array import hstack
from chainer.functions.array import permutate
from chainer.functions.array import reshape
from chainer.functions.array import rollaxis
from chainer.functions.array import select_item
from chainer.functions.array import separate
from chainer.functions.array import split_axis
from chainer.functions.array import stack
from chainer.functions.array import swapaxes
from chainer.functions.array import transpose
from chainer.functions.array import transpose_sequence
from chainer.functions.array import vstack
from chainer.functions.array import where
from chainer.functions.connection import bilinear
from chainer.functions.connection import convolution_2d
from chainer.functions.connection import convolution_nd
from chainer.functions.connection import deconvolution_2d
from chainer.functions.connection import deconvolution_nd
from chainer.functions.connection import dilated_convolution_2d
from chainer.functions.connection import embed_id
from chainer.functions.connection import linear
from chainer.functions.connection import n_step_lstm
from chainer.functions.evaluation import accuracy
from chainer.functions.evaluation import binary_accuracy
from chainer.functions.evaluation import classification_summary \
    as classification_summary_
from chainer.functions.loss import black_out
from chainer.functions.loss import contrastive
from chainer.functions.loss import crf1d
from chainer.functions.loss import cross_covariance
from chainer.functions.loss import ctc
from chainer.functions.loss import hinge
from chainer.functions.loss import huber_loss
from chainer.functions.loss import mean_squared_error
from chainer.functions.loss import negative_sampling
from chainer.functions.loss import sigmoid_cross_entropy
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.loss import triplet
from chainer.functions.loss import vae  # NOQA
from chainer.functions.math import basic_math  # NOQA
from chainer.functions.math import batch_l2_norm_squared
from chainer.functions.math import bias
from chainer.functions.math import ceil
from chainer.functions.math import clip
from chainer.functions.math import det
from chainer.functions.math import exponential
from chainer.functions.math import exponential_m1
from chainer.functions.math import floor
from chainer.functions.math import hyperbolic
from chainer.functions.math import identity
from chainer.functions.math import inv
from chainer.functions.math import linear_interpolate
from chainer.functions.math import logarithm_1p
from chainer.functions.math import logsumexp
from chainer.functions.math import matmul
from chainer.functions.math import maximum
from chainer.functions.math import minimum
from chainer.functions.math import minmax
from chainer.functions.math import scale
from chainer.functions.math import sqrt
from chainer.functions.math import sum
from chainer.functions.math import trigonometric
from chainer.functions.noise import dropout
from chainer.functions.noise import gaussian
from chainer.functions.normalization import batch_normalization
from chainer.functions.normalization import l2_normalization
from chainer.functions.normalization import local_response_normalization
from chainer.functions.pooling import average_pooling_2d
from chainer.functions.pooling import max_pooling_2d
from chainer.functions.pooling import roi_pooling_2d
from chainer.functions.pooling import spatial_pyramid_pooling_2d
from chainer.functions.pooling import unpooling_2d
from chainer.functions.util import forget
from chainer.links.activation import prelu as links_prelu
from chainer.links.connection import bilinear as links_bilinear
from chainer.links.connection import convolution_2d as links_convolution_2d
from chainer.links.connection import dilated_convolution_2d \
    as links_dilated_convolution_2d
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
CReLU = crelu.CReLU
crelu = crelu.crelu
ConnectionistTemporalClassification = ctc.ConnectionistTemporalClassification
connectionist_temporal_classification \
    = ctc.connectionist_temporal_classification
ELU = elu.ELU
elu = elu.elu
HardSigmoid = hard_sigmoid.HardSigmoid
hard_sigmoid = hard_sigmoid.hard_sigmoid
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
Cast = cast.Cast
cast = cast.cast
Concat = concat.Concat
concat = concat.concat
Copy = copy.Copy
copy = copy.copy
dstack = dstack.dstack
ExpandDims = expand_dims.ExpandDims
expand_dims = expand_dims.expand_dims
Flatten = flatten.Flatten
flatten = flatten.flatten
GetItem = get_item.GetItem
get_item = get_item.get_item
hstack = hstack.hstack
Permutate = permutate.Permutate
permutate = permutate.permutate
Reshape = reshape.Reshape
reshape = reshape.reshape
Rollaxis = rollaxis.Rollaxis
rollaxis = rollaxis.rollaxis
SelectItem = select_item.SelectItem
select_item = select_item.select_item
separate = separate.separate
SplitAxis = split_axis.SplitAxis
split_axis = split_axis.split_axis
stack = stack.stack
Swapaxes = swapaxes.Swapaxes
swapaxes = swapaxes.swapaxes
Transpose = transpose.Transpose
transpose = transpose.transpose
TransposeSequence = transpose_sequence.TransposeSequence
transpose_sequence = transpose_sequence.transpose_sequence
Where = where.Where
where = where.where

bilinear = bilinear.bilinear
convolution_2d = convolution_2d.convolution_2d
convolution_nd = convolution_nd.convolution_nd
deconvolution_2d = deconvolution_2d.deconvolution_2d
deconvolution_nd = deconvolution_nd.deconvolution_nd
dilated_convolution_2d = dilated_convolution_2d.dilated_convolution_2d
embed_id = embed_id.embed_id
linear = linear.linear
NStepLSTM = n_step_lstm.NStepLSTM
n_step_lstm = n_step_lstm.n_step_lstm

Accuracy = accuracy.Accuracy
accuracy = accuracy.accuracy
BinaryAccuracy = binary_accuracy.BinaryAccuracy
binary_accuracy = binary_accuracy.binary_accuracy
ClassificationSummary = classification_summary_.ClassificationSummary
classification_summary = classification_summary_.classification_summary
precision = classification_summary_.precision
recall = classification_summary_.recall
f1_score = classification_summary_.f1_score

bernoulli_nll = vae.bernoulli_nll
BinaryHierarchicalSoftmax = hierarchical_softmax.BinaryHierarchicalSoftmax
black_out = black_out.black_out
Contrastive = contrastive.Contrastive
contrastive = contrastive.contrastive
crf1d = crf1d.crf1d
CrossCovariance = cross_covariance.CrossCovariance
cross_covariance = cross_covariance.cross_covariance
gaussian_kl_divergence = vae.gaussian_kl_divergence
gaussian_nll = vae.gaussian_nll
Hinge = hinge.Hinge
hinge = hinge.hinge
HuberLoss = huber_loss.HuberLoss
huber_loss = huber_loss.huber_loss
MeanSquaredError = mean_squared_error.MeanSquaredError
mean_squared_error = mean_squared_error.mean_squared_error
negative_sampling = negative_sampling.negative_sampling
SigmoidCrossEntropy = sigmoid_cross_entropy.SigmoidCrossEntropy
sigmoid_cross_entropy = sigmoid_cross_entropy.sigmoid_cross_entropy
SoftmaxCrossEntropy = softmax_cross_entropy.SoftmaxCrossEntropy
softmax_cross_entropy = softmax_cross_entropy.softmax_cross_entropy
Triplet = triplet.Triplet
triplet = triplet.triplet
vstack = vstack.vstack

ArgMax = minmax.ArgMax
argmax = minmax.argmax
ArgMin = minmax.ArgMin
argmin = minmax.argmin
Arccos = trigonometric.Arccos
arccos = trigonometric.arccos
Arcsin = trigonometric.Arcsin
arcsin = trigonometric.arcsin
Arctan = trigonometric.Arctan
arctan = trigonometric.arctan
BatchDet = det.BatchDet
batch_det = det.batch_det
BatchInv = inv.BatchInv
batch_inv = inv.batch_inv
BatchL2NormSquared = batch_l2_norm_squared.BatchL2NormSquared
batch_l2_norm_squared = batch_l2_norm_squared.batch_l2_norm_squared
BatchMatMul = matmul.BatchMatMul
batch_matmul = matmul.batch_matmul
bias = bias.bias
Ceil = ceil.Ceil
ceil = ceil.ceil
Clip = clip.Clip
clip = clip.clip
Cos = trigonometric.Cos
cos = trigonometric.cos
Cosh = hyperbolic.Cosh
cosh = hyperbolic.cosh
det = det.det
Exp = exponential.Exp
exp = exponential.exp
Expm1 = exponential_m1.Expm1
expm1 = exponential_m1.expm1
Floor = floor.Floor
floor = floor.floor
Identity = identity.Identity
identity = identity.identity
Inv = inv.Inv
inv = inv.inv
LinearInterpolate = linear_interpolate.LinearInterpolate
linear_interpolate = linear_interpolate.linear_interpolate
Log = exponential.Log
log = exponential.log
Log10 = exponential.Log10
log10 = exponential.log10
Log1p = logarithm_1p.Log1p
log1p = logarithm_1p.log1p
Log2 = exponential.Log2
log2 = exponential.log2
LogSumExp = logsumexp.LogSumExp
logsumexp = logsumexp.logsumexp
MatMul = matmul.MatMul
matmul = matmul.matmul
Max = minmax.Max
max = minmax.max
Maximum = maximum.Maximum
maximum = maximum.maximum
Minimum = minimum.Minimum
minimum = minimum.minimum
Min = minmax.Min
min = minmax.min
rsqrt = sqrt.rsqrt
scale = scale.scale
Sin = trigonometric.Sin
sin = trigonometric.sin
Sinh = hyperbolic.Sinh
sinh = hyperbolic.sinh
Sqrt = sqrt.Sqrt
sqrt = sqrt.sqrt
Sum = sum.Sum
sum = sum.sum
Tan = trigonometric.Tan
tan = trigonometric.tan

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
NormalizeL2 = l2_normalization.NormalizeL2
normalize = l2_normalization.normalize

AveragePooling2D = average_pooling_2d.AveragePooling2D
average_pooling_2d = average_pooling_2d.average_pooling_2d
MaxPooling2D = max_pooling_2d.MaxPooling2D
max_pooling_2d = max_pooling_2d.max_pooling_2d
ROIPooling2D = roi_pooling_2d.ROIPooling2D
roi_pooling_2d = roi_pooling_2d.roi_pooling_2d
SpatialPyramidPooling2D = spatial_pyramid_pooling_2d.SpatialPyramidPooling2D
spatial_pyramid_pooling_2d = \
    spatial_pyramid_pooling_2d.spatial_pyramid_pooling_2d

Unpooling2D = unpooling_2d.Unpooling2D
unpooling_2d = unpooling_2d.unpooling_2d

Forget = forget.Forget
forget = forget.forget

# Import for backward compatibility
PReLU = links_prelu.PReLU

Bilinear = links_bilinear.Bilinear
Convolution2D = links_convolution_2d.Convolution2D
DilatedConvolution2D = links_dilated_convolution_2d.DilatedConvolution2D
EmbedID = links_embed_id.EmbedID
Inception = inception.Inception
InceptionBN = inceptionbn.InceptionBN
Linear = links_linear.Linear
Parameter = parameter.Parameter

NegativeSampling = links_negative_sampling.NegativeSampling

BatchNormalization = links_batch_normalization.BatchNormalization
