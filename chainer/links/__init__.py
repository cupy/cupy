"""Collection of :class:`~chainer.Link` implementations."""

from chainer.links.activation import prelu
from chainer.links.connection import bilinear
from chainer.links.connection import convolution_2d
from chainer.links.connection import embed_id
from chainer.links.connection import inception
from chainer.links.connection import inceptionbn
from chainer.links.connection import linear
from chainer.links.connection import parameter
from chainer.links.loss import hierarchical_softmax
from chainer.links.loss import negative_sampling
from chainer.links.normalization import batch_normalization


PReLU = prelu.PReLU

Bilinear = bilinear.Bilinear
Convolution2D = convolution_2d.Convolution2D
EmbedID = embed_id.EmbedID
Inception = inception.Inception
InceptionBN = inceptionbn.InceptionBN
Linear = linear.Linear
Parameter = parameter.Parameter

BinaryHierarchicalSoftmax = hierarchical_softmax.BinaryHierarchicalSoftmax
NegativeSampling = negative_sampling.NegativeSampling

BatchNormalization = batch_normalization.BatchNormalization
