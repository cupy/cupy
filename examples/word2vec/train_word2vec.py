#!/usr/bin/env python
"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
Use ../ptb/download.py to download 'ptb.train.txt'.
"""
import argparse
import collections
import time

import numpy as np

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
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--epoch', '-e', default=10, type=int,
                    help='number of epochs to learn')
parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'],
                    default='skipgram',
                    help='model type ("skipgram", "cbow")')
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
        h = F.sum(e, axis=0)
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
        dummy = chainer.Variable(
            xp.empty((shape[0], shape[1])))
        x, _ = F.broadcast(x, dummy)
        e = F.reshape(e, (shape[0] * shape[1], shape[2]))
        x = F.reshape(x, (shape[0] * shape[1],))
        return self.loss_func(e, x)


class SoftmaxCrossEntropyLoss(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropyLoss, self).__init__(
            W=L.Linear(n_in, n_out),
        )

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.W(x), t)


def calculate_loss(model, dataset, position):
    # use random window size in the same way as the original word2vec
    # implementation.
    w = np.random.randint(args.window - 1) + 1
    pos = [position + o for o in range(-w, w + 1) if o != 0]
    d = xp.asarray(np.take(dataset, pos))
    context = chainer.Variable(d)
    x_data = xp.asarray(np.take(dataset, position))
    x = chainer.Variable(x_data)
    return model(x, context)


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()

index2word = {}
word2index = {}
counts = collections.Counter()
dataset = []
with open('ptb.train.txt') as f:
    for line in f:
        for word in line.split():
            if word not in word2index:
                ind = len(word2index)
                word2index[word] = ind
                index2word[ind] = word
            counts[word2index[word]] += 1
            dataset.append(word2index[word])
            if args.test and len(dataset) >= 100:
                break
        if args.test and len(dataset) >= 100:
            break

n_vocab = len(word2index)

print('n_vocab: %d' % n_vocab)
print('data length: %d' % len(dataset))

if args.out_type == 'hsm':
    HSM = L.BinaryHierarchicalSoftmax
    tree = HSM.create_huffman_tree(counts)
    loss_func = HSM(args.unit, tree)
elif args.out_type == 'ns':
    cs = [counts[w] for w in range(len(counts))]
    loss_func = L.NegativeSampling(args.unit, cs, 20)
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

if args.gpu >= 0:
    model.to_gpu()

dataset = np.array(dataset, dtype=np.int32)

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
    for i in indexes:
        if word_count >= next_count:
            now = time.time()
            duration = now - cur_at
            throuput = 100000. / (now - cur_at)
            print('{} words, {:.2f} sec, {:.2f} words/sec'.format(
                word_count, duration, throuput))
            next_count += 100000
            cur_at = now

        position = np.array(
            range(0, args.batchsize)) * skip + (args.window + i)
        loss = calculate_loss(model, dataset, position)
        accum_loss += loss.data
        word_count += args.batchsize

        model.zerograds()
        loss.backward()
        del loss
        optimizer.update()

    print(accum_loss)

with open('word2vec.model', 'w') as f:
    f.write('%d %d\n' % (len(index2word), args.unit))
    w = model.embed.W.data
    for i in range(w.shape[0]):
        v = ' '.join(['%f' % v for v in w[i]])
        f.write('%s %s\n' % (index2word[i], v))
