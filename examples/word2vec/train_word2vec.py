#!/usr/bin/env python
"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
Use ../ptb/download.py to download 'ptb.train.txt'.
"""
import argparse
import collections
import time

import numpy as np
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
import chainer.functions as F
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
args = parser.parse_args()

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('Window: {}'.format(args.window))
print('Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Training model: {}'.format(args.model))
print('Output type: {}'.format(args.out_type))
print('')


def continuous_bow(dataset, position):
    h = None

    # use random window size in the same way as the original word2vec
    # implementation.
    w = np.random.randint(args.window - 1) + 1
    for offset in range(-w, w + 1):
        if offset == 0:
            continue
        d = dataset[position + offset]
        if args.gpu >= 0:
            d = cuda.to_gpu(d)
        x = chainer.Variable(d)
        e = model.embed(x)
        h = h + e if h is not None else e

    d = dataset[position]
    if args.gpu >= 0:
        d = cuda.to_gpu(d)
    t = chainer.Variable(d)
    return loss_func(h, t)


def skip_gram(dataset, position):
    d = dataset[position]
    if args.gpu >= 0:
        d = cuda.to_gpu(d)
    t = chainer.Variable(d)

    # use random window size in the same way as the original word2vec
    # implementation.
    w = np.random.randint(args.window - 1) + 1
    loss = None
    for offset in range(-w, w + 1):
        if offset == 0:
            continue
        d = dataset[position + offset]
        if args.gpu >= 0:
            d = cuda.to_gpu(d)
        x = chainer.Variable(d)
        e = model.embed(x)

        loss_i = loss_func(e, t)
        loss = loss_i if loss is None else loss + loss_i

    return loss


if args.gpu >= 0:
    cuda.init(args.gpu)

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

n_vocab = len(word2index)

print('n_vocab: %d' % n_vocab)
print('data length: %d' % len(dataset))

if args.model == 'skipgram':
    train_model = skip_gram
elif args.model == 'cbow':
    train_model = continuous_bow
else:
    raise Exception('Unknown model type: {}'.format(args.model))

model = chainer.FunctionSet(
    embed=F.EmbedID(n_vocab, args.unit),
)

if args.out_type == 'hsm':
    tree = F.create_huffman_tree(counts)
    model.l = F.BinaryHierarchicalSoftmax(args.unit, tree)
    loss_func = model.l
elif args.out_type == 'ns':
    cs = [counts[w] for w in range(len(counts))]
    model.l = F.NegativeSampling(args.unit, cs, 20)
    loss_func = model.l
elif args.out_type == 'original':
    model.l = F.Linear(args.unit, n_vocab)
    loss_func = lambda h, t: F.softmax_cross_entropy(model.l(h), t)
else:
    raise Exception('Unknown output type: {}'.format(args.out_type))

if args.gpu >= 0:
    model.to_gpu()

dataset = np.array(dataset, dtype=np.int32)

optimizer = O.Adam()
optimizer.setup(model.collect_parameters())

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
        loss = train_model(dataset, position)
        accum_loss += loss.data
        word_count += args.batchsize

        optimizer.zero_grads()
        loss.backward()
        optimizer.update()

    print(accum_loss)

model.to_cpu()
with open('model.pickle', 'wb') as f:
    obj = (model, index2word, word2index)
    pickle.dump(obj, f)
