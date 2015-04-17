#!/usr/bin/env python
"""Sample script to train recurrent networks language model.

For more information, refer to reference codes written in Torch:
https://github.com/tomsercu/lstm
"""

import cPickle as pickle
import datetime, math, sys, time

import numpy as np
from pycuda import gpuarray

from chainer import Variable, FunctionSet
from chainer.functions import dropout, EmbedID, Linear, lstm, softmax_cross_entropy
from chainer.optimizers import SGD

# Prepare dataset
vocab = {}
n_vocab = 0
def load_data(filename):
    global vocab, n_vocab
    txt = open(filename).read()
    words = txt.replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = n_vocab
            n_vocab += 1
        dataset[i] = vocab[word]
    return dataset

train_data = load_data('ptb.train.txt')
valid_data = load_data('ptb.valid.txt')
test_data  = load_data('ptb.test.txt')
print 'n_vocab =', n_vocab

# Initialize model
n_unit = 200
model = FunctionSet(
    embed=EmbedID(n_vocab, n_unit),
    l1_x=Linear(n_unit, 4*n_unit),
    l1_h=Linear(n_unit, 4*n_unit),
    l2_x=Linear(n_unit, 4*n_unit),
    l2_h=Linear(n_unit, 4*n_unit),
    l3  =Linear(n_unit, n_vocab)
)
model.to_gpu()

# Setup optimizer
optimizer = SGD(lr=1.)
optimizer.setup(model.collect_parameters())

# Architecture
def forward_one_step(x_data, y_data, state, train=True):
    x = Variable(x_data, volatile=not train)
    t = Variable(y_data, volatile=not train)
    h0     = model.embed(x)
    h1_in  = model.l1_x(h0) + model.l1_h(state['h1'])
    c1, h1 = lstm(state['c1'], dropout(h1_in, train=train))
    h2_in  = model.l2_x(h1) + model.l2_h(state['h2'])
    c2, h2 = lstm(state['c2'], dropout(h2_in, train=train))
    y      = model.l3(h2)
    state  = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    return state, softmax_cross_entropy(y, t)

def make_initial_state(volatile=False):
    return {name: Variable(gpuarray.zeros((batchsize, n_unit), dtype=np.float32,
                                          volatile=volatile))
            for name in ('c1', 'h1', 'c2', 'h2')}

whole_len = train_data.shape[0]
batchsize = 20
bpsize    = 20
grad_clip = 5
showloss  = 10000
def train_one_epoch(state):
    jump = whole_len / batchsize
    cur_log_perp = 0
    sum_log_perp = 0

    start_at = time.time()
    cur_at   = start_at
    for i in xrange(whole_len):
        x_cpu = np.array([train_data[(jump * j + i) % whole_len]
                          for j in xrange(batchsize)])
        y_cpu = np.array([train_data[(jump * j + i + 1) % whole_len]
                          for j in xrange(batchsize)])
        state, loss = forward_one_step(
            gpuarray.to_gpu(x_cpu), gpuarray.to_gpu(y_cpu), state)
        log_perp = float(loss.data.get()) / batchsize
        cur_log_perp += log_perp
        sum_log_perp += log_perp

        if (i + 1) % bpsize == 0:
            optimizer.zero_grads()
            loss.backward()
            loss.forget_backward()
            optimizer.clip_grads(grad_clip)
            optimizer.update()

        if (i + 1) % showloss == 0:
            perp = math.exp(cur_log_perp / showloss)
            print '\ncurrent perplexity:', perp
            cur_log_perp = 0

            now = time.time()
            print 'duration:', datetime.timedelta(seconds = now - cur_at)
            cur_at = now

        sys.stdout.write('\rtrain: {} / {}'.format(i + 1, whole_len))
        sys.stdout.flush()

    print '\nwhole time:', datetime.timedelta(seconds = time.time() - start_at)
    return state, sum_log_perp / whole_len

def evaluate(dataset):
    dataset_len = dataset.size
    sum_log_perp = 0
    state = make_initial_state(volatile=True)

    start_at = time.time()
    for i in xrange(dataset_len - 1):
        sys.stdout.write('\reval: {} / {}'.format(i, dataset_len - 1))
        sys.stdout.flush()

        x_cpu = dataset[i   : i+1]
        y_cpu = dataset[i+1 : i+2]
        state, loss = forward_one_step(
            gpuarray.to_gpu(x_cpu), gpuarray.to_gpu(y_cpu), state, train=False)
        sum_log_perp += float(loss.data.get())

    print '\nwhole time:', datetime.timedelta(seconds = time.time() - start_at)
    return sum_log_perp / (dataset_len - 1)

n_epoch = 1
def train_whole():
    state = make_initial_state()
    start_at = time.time()
    for epoch in xrange(n_epoch):
        print 'epoch', n_epoch
        print 'train'
        state, train_log_perp = train_one_epoch(state)
        print 'train perplexity:     ', math.exp(train_log_perp)
        print 'validate'
        valid_log_perp = evaluate(valid_data)
        print 'validation perplexity:', math.exp(valid_log_perp)

    print 'test'
    test_log_perp = evaluate(test_data)
    print 'test perplexity:      ', math.exp(test_log_perp)
    print '\nwhole time:', datetime.timedelta(seconds = time.time() - start_at)

train_whole()

with open('model', 'wb') as f:
    pickle.dump(model, f, -1)
