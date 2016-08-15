#!/usr/bin/env python
"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
Use ../ptb/download.py to download 'ptb.train.txt'.
"""
import argparse
import collections
import time

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--unit', '-u', default=100, type=int,
                    help='number of units')
parser.add_argument('--window', '-w', default=5, type=int,
                    help='window size')
parser.add_argument('--batchsize', '-b', type=int, default=1000,
                    help='learning minibatch size')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'],
                    default='skipgram',
                    help='model type ("skipgram", "cbow")')
parser.add_argument('--negative-size', default=5, type=int,
                    help='number of negative samples')
parser.add_argument('--out-type', '-o', choices=['hsm', 'ns', 'original'],
                    default='hsm',
                    help='output model type ("hsm": hierarchical softmax, '
                    '"ns": negative sampling, "original": no approximation)')
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)

args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('Window: {}'.format(args.window))
print('Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Training model: {}'.format(args.model))
print('Output type: {}'.format(args.out_type))
print('')


class ContinuousBoW(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(ContinuousBoW, self).__init__(
            embed=F.EmbedID(n_vocab, n_units),
            loss_func=loss_func,
        )

    def __call__(self, x, context):
        e = model.embed(context)
        h = F.sum(e, axis=0) * (1. / len(context.data))
        return self.loss_func(h, x)


class SkipGram(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(SkipGram, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            loss_func=loss_func,
        )

    def __call__(self, x, context):
        e = model.embed(context)
        shape = e.data.shape
        x = F.broadcast_to(x, (shape[0], shape[1]))
        e = F.reshape(e, (shape[0] * shape[1], shape[2]))
        x = F.reshape(x, (shape[0] * shape[1],))
        return self.loss_func(e, x)


class SoftmaxCrossEntropyLoss(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropyLoss, self).__init__(
            out=L.Linear(n_in, n_out),
        )
        self.out.W.data[...] = 0

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.out(x), t)


def calculate_loss(model, dataset, position):
    # use random window size in the same way as the original word2vec
    # implementation.
    w = np.random.randint(args.window - 1) + 1
    # offset is [-w, ..., -1, 1, ..., w]
    offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
    pos = np.expand_dims(position, 0) + np.expand_dims(offset, 1)
    d = xp.asarray(dataset.take(pos))
    context = chainer.Variable(d)
    x_data = xp.asarray(dataset.take(position))
    x = chainer.Variable(x_data)
    return model(x, context)


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()

train, _, _ = chainer.datasets.get_ptb_words()
if args.test:
    train = train[:100]

vocab = chainer.datasets.get_ptb_words_vocabulary()
index2word = {wid: word for word, wid in six.iteritems(vocab)}

counts = collections.Counter(train)
n_vocab = max(train) + 1

print('n_vocab: %d' % n_vocab)
print('data length: %d' % len(train))

if args.out_type == 'hsm':
    HSM = L.BinaryHierarchicalSoftmax
    tree = HSM.create_huffman_tree(counts)
    loss_func = HSM(args.unit, tree)
    loss_func.W.data[...] = 0
elif args.out_type == 'ns':
    cs = [counts[w] for w in range(len(counts))]
    loss_func = L.NegativeSampling(args.unit, cs, args.negative_size)
    loss_func.W.data[...] = 0
elif args.out_type == 'original':
    loss_func = SoftmaxCrossEntropyLoss(args.unit, n_vocab)
else:
    raise Exception('Unknown output type: {}'.format(args.out_type))

if args.model == 'skipgram':
    model = SkipGram(n_vocab, args.unit, loss_func)
elif args.model == 'cbow':
    model = ContinuousBoW(n_vocab, args.unit, loss_func)
else:
    raise Exception('Unknown model type: {}'.format(args.model))

model.embed.W.data[...] = np.random.uniform(-1, 1, (n_vocab, args.unit)) \
                                   .astype(np.float32) / args.unit

if args.gpu >= 0:
    model.to_gpu()

dataset = np.array(train, dtype=np.int32)

optimizer = O.Adam()
optimizer.setup(model)

begin_time = time.time()
cur_at = begin_time
word_count = 0
skip = (len(dataset) - args.window * 2) // args.batchsize
next_count = 100000
for epoch in range(args.epoch):
    accum_loss = 0
    print('epoch: {0}'.format(epoch))
    indexes = np.random.permutation(skip)
    position = np.arange(0, args.batchsize * skip, skip) + args.window
    for i in indexes:
        if word_count >= next_count:
            now = time.time()
            duration = now - cur_at
            throuput = 100000. / (now - cur_at)
            print('{} words, {:.2f} sec, {:.2f} Kwords/sec'.format(
                word_count, duration, throuput / 1000.))
            next_count += 100000
            cur_at = now

        loss = calculate_loss(model, dataset, position)
        accum_loss += loss.data
        word_count += args.batchsize

        model.zerograds()
        loss.backward()
        del loss
        optimizer.update()

        position += 1

    print(accum_loss)

with open('word2vec.model', 'w') as f:
    f.write('%d %d\n' % (len(index2word), args.unit))
    w = cuda.to_cpu(model.embed.W.data)
    for i, wi in enumerate(w):
        v = ' '.join(map(str, wi))
        f.write('%s %s\n' % (index2word[i], v))
