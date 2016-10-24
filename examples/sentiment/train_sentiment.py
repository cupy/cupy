#!/usr/bin/env python
"""Sample script of recursive neural networks for sentiment analysis.

This is Socher's simple recursive model, not RTNN:
  R. Socher, C. Lin, A. Y. Ng, and C.D. Manning.
  Parsing Natural Scenes and Natural Language with Recursive Neural Networks.
  in ICML2011.

"""

import argparse
import codecs
import collections
import random
import re
import time

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=400, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=30, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=25,
                    help='learning minibatch size')
parser.add_argument('--label', '-l', type=int, default=5,
                    help='number of labels')
parser.add_argument('--epocheval', '-p', type=int, default=5,
                    help='number of epochs per evaluation')
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.epoch       # number of epochs
n_units = args.unit        # number of units per layer
batchsize = args.batchsize      # minibatch size
n_label = args.label         # number of labels
epoch_per_eval = args.epocheval  # number of epochs per evaluation


class SexpParser(object):

    def __init__(self, line):
        self.tokens = re.findall(r'\(|\)|[^\(\) ]+', line)
        self.pos = 0

    def parse(self):
        assert self.pos < len(self.tokens)
        token = self.tokens[self.pos]
        assert token != ')'
        self.pos += 1

        if token == '(':
            children = []
            while True:
                assert self.pos < len(self.tokens)
                if self.tokens[self.pos] == ')':
                    self.pos += 1
                    break
                else:
                    children.append(self.parse())
            return children
        else:
            return token


def convert_tree(vocab, exp):
    assert isinstance(exp, list) and (len(exp) == 2 or len(exp) == 3)

    if len(exp) == 2:
        label, leaf = exp
        if leaf not in vocab:
            vocab[leaf] = len(vocab)
        return {'label': int(label), 'node': vocab[leaf]}
    elif len(exp) == 3:
        label, left, right = exp
        node = (convert_tree(vocab, left), convert_tree(vocab, right))
        return {'label': int(label), 'node': node}


def read_corpus(path, vocab, max_size):
    with codecs.open(path, encoding='utf-8') as f:
        trees = []
        for line in f:
            line = line.strip()
            tree = SexpParser(line).parse()
            trees.append(convert_tree(vocab, tree))
            if max_size and len(trees) >= max_size:
                break

        return trees


class RecursiveNet(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        super(RecursiveNet, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l=L.Linear(n_units * 2, n_units),
            w=L.Linear(n_units, n_label))

    def leaf(self, x):
        return self.embed(x)

    def node(self, left, right):
        return F.tanh(self.l(F.concat((left, right))))

    def label(self, v):
        return self.w(v)


def traverse(model, node, train=True, evaluate=None, root=True):
    if isinstance(node['node'], int):
        # leaf node
        word = xp.array([node['node']], np.int32)
        loss = 0
        x = chainer.Variable(word, volatile=not train)
        v = model.leaf(x)
    else:
        # internal node
        left_node, right_node = node['node']
        left_loss, left = traverse(
            model, left_node, train=train, evaluate=evaluate, root=False)
        right_loss, right = traverse(
            model, right_node, train=train, evaluate=evaluate, root=False)
        v = model.node(left, right)
        loss = left_loss + right_loss

    y = model.label(v)

    if train:
        label = xp.array([node['label']], np.int32)
        t = chainer.Variable(label, volatile=not train)
        loss += F.softmax_cross_entropy(y, t)

    if evaluate is not None:
        predict = cuda.to_cpu(y.data.argmax(1))
        if predict[0] == node['label']:
            evaluate['correct_node'] += 1
        evaluate['total_node'] += 1

        if root:
            if predict[0] == node['label']:
                evaluate['correct_root'] += 1
            evaluate['total_root'] += 1

    return loss, v


def evaluate(model, test_trees):
    m = model.copy()
    m.volatile = True
    result = collections.defaultdict(lambda: 0)
    for tree in test_trees:
        traverse(m, tree, train=False, evaluate=result)

    acc_node = 100.0 * result['correct_node'] / result['total_node']
    acc_root = 100.0 * result['correct_root'] / result['total_root']
    print(' Node accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_node, result['correct_node'], result['total_node']))
    print(' Root accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_root, result['correct_root'], result['total_root']))

vocab = {}
if args.test:
    max_size = 10
else:
    max_size = None
train_trees = read_corpus('trees/train.txt', vocab, max_size)
test_trees = read_corpus('trees/test.txt', vocab, max_size)
develop_trees = read_corpus('trees/dev.txt', vocab, max_size)

model = RecursiveNet(len(vocab), n_units)

if args.gpu >= 0:
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.AdaGrad(lr=0.1)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

accum_loss = 0
count = 0
start_at = time.time()
cur_at = start_at
for epoch in range(n_epoch):
    print('Epoch: {0:d}'.format(epoch))
    total_loss = 0
    cur_at = time.time()
    random.shuffle(train_trees)
    for tree in train_trees:
        loss, v = traverse(model, tree, train=True)
        accum_loss += loss
        count += 1

        if count >= batchsize:
            model.cleargrads()
            accum_loss.backward()
            optimizer.update()
            total_loss += float(accum_loss.data)

            accum_loss = 0
            count = 0

    print('loss: {:.2f}'.format(total_loss))

    now = time.time()
    throughput = float(len(train_trees)) / (now - cur_at)
    print('{:.2f} iters/sec, {:.2f} sec'.format(throughput, now - cur_at))
    print()

    if (epoch + 1) % epoch_per_eval == 0:
        print('Train data evaluation:')
        evaluate(model, train_trees)
        print('Develop data evaluation:')
        evaluate(model, develop_trees)
        print('')

print('Test evaluateion')
evaluate(model, test_trees)
