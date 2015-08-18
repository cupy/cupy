#!/usr/bin/env python
"""Sample script of recursive neural networks for sentiment analysis.

This is Socher's simple recursive model, not RTNN:
  R. Socher, C. Lin, A. Y. Ng, and C.D. Manning.
  Parsing Natural Scenes and Natural Language with Recursive Neural Networks.
  in ICML2011.

"""

import argparse
import collections
import random
import re
import time

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

n_epoch = 400       # number of epochs
n_units = 30        # number of units per layer
batchsize = 25      # minibatch size
n_label = 5         # number of labels
epoch_per_eval = 5  # number of epochs per evaluation


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


def read_corpus(path, vocab):
    with open(path) as f:
        trees = []
        for line in f:
            line = line.strip()
            tree = SexpParser(line).parse()
            trees.append(convert_tree(vocab, tree))

        return trees


def traverse(node, train=True, evaluate=None, root=True):
    if isinstance(node['node'], int):
        # leaf node
        word = np.array([node['node']], np.int32)
        if args.gpu >= 0:
            word = cuda.to_gpu(word)
        loss = 0
        x = chainer.Variable(word, volatile=not train)
        v = model.embed(x)
    else:
        # internal node
        left_node, right_node = node['node']
        left_loss, left = traverse(
            left_node, train=train, evaluate=evaluate, root=False)
        right_loss, right = traverse(
            right_node, train=train, evaluate=evaluate, root=False)
        v = F.tanh(model.l(F.concat((left, right))))
        loss = left_loss + right_loss

    y = model.w(v)

    if train:
        label = np.array([node['label']], np.int32)
        if args.gpu >= 0:
            label = cuda.to_gpu(label)
        t = chainer.Variable(label, volatile=not train)
        loss += F.softmax_cross_entropy(y, t)

    if evaluate is not None:
        predict = cuda.to_cpu(y.data).argmax(1)
        if predict[0] == node['label']:
            evaluate['correct_node'] += 1
        evaluate['total_node'] += 1

        if root:
            if predict[0] == node['label']:
                evaluate['correct_root'] += 1
            evaluate['total_root'] += 1

    return loss, v


def evaluate(test_trees):
    result = collections.defaultdict(lambda: 0)
    for tree in test_trees:
        traverse(tree, train=False, evaluate=result)

    acc_node = 100.0 * result['correct_node'] / result['total_node']
    acc_root = 100.0 * result['correct_root'] / result['total_root']
    print(' Node accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_node, result['correct_node'], result['total_node']))
    print(' Root accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_root, result['correct_root'], result['total_root']))

vocab = {}
train_trees = read_corpus('trees/train.txt', vocab)
test_trees = read_corpus('trees/test.txt', vocab)
develop_trees = read_corpus('trees/dev.txt', vocab)

model = chainer.FunctionSet(
    embed=F.EmbedID(len(vocab), n_units),
    l=F.Linear(n_units * 2, n_units),
    w=F.Linear(n_units, n_label),
)

if args.gpu >= 0:
    cuda.init()
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.AdaGrad(lr=0.1)
optimizer.setup(model)

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
        loss, v = traverse(tree, train=True)
        accum_loss += loss
        count += 1

        if count >= batchsize:
            optimizer.zero_grads()
            accum_loss.backward()
            optimizer.weight_decay(0.0001)
            optimizer.update()
            total_loss += float(cuda.to_cpu(accum_loss.data))

            accum_loss = 0
            count = 0

    print('loss: {:.2f}'.format(total_loss))

    now = time.time()
    throuput = float(len(train_trees)) / (now - cur_at)
    print('{:.2f} iters/sec, {:.2f} sec'.format(throuput, now - cur_at))
    print()

    if (epoch + 1) % epoch_per_eval == 0:
        print('Train data evaluation:')
        evaluate(train_trees)
        print('Develop data evaluation:')
        evaluate(develop_trees)
        print('')

print('Test evaluateion')
evaluate(test_trees)
