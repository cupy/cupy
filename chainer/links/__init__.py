"""Collection of :class:`~chainer.Link` implementations."""

from chainer.links.activation import maxout
from chainer.links.activation import prelu
from chainer.links.connection import bilinear
from chainer.links.connection import convolution_2d
from chainer.links.connection import deconvolution_2d
from chainer.links.connection import embed_id
from chainer.links.connection import gru
from chainer.links.connection import inception
from chainer.links.connection import inceptionbn
from chainer.links.connection import linear
from chainer.links.connection import lstm
from chainer.links.connection import mlp_convolution_2d
from chainer.links.connection import parameter
from chainer.links.loss import hierarchical_softmax
from chainer.links.loss import negative_sampling
from chainer.links.model import classifier
from chainer.links.normalization import batch_normalization


Maxout = maxout.Maxout
PReLU = prelu.PReLU

Bilinear = bilinear.Bilinear
Convolution2D = convolution_2d.Convolution2D
Deconvolution2D = deconvolution_2d.Deconvolution2D
EmbedID = embed_id.EmbedID
GRU = gru.GRU
StatefulGRU = gru.StatefulGRU
Inception = inception.Inception
InceptionBN = inceptionbn.InceptionBN
Linear = linear.Linear
LSTM = lstm.LSTM
MLPConvolution2D = mlp_convolution_2d.MLPConvolution2D
Parameter = parameter.Parameter

BinaryHierarchicalSoftmax = hierarchical_softmax.BinaryHierarchicalSoftmax
NegativeSampling = negative_sampling.NegativeSampling

Classifier = classifier.Classifier

BatchNormalization = batch_normalization.BatchNormalization
