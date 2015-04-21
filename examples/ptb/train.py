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
n_unit = 650
model = FunctionSet(
    embed=EmbedID(n_vocab, n_unit),
    l1_x=Linear(n_unit, 4*n_unit),
    l1_h=Linear(n_unit, 4*n_unit),
    l2_x=Linear(n_unit, 4*n_unit),
    l2_h=Linear(n_unit, 4*n_unit),
    l3  =Linear(n_unit, n_vocab)
)
for p in model.collect_parameters()[0]:
    p[:] = np.random.uniform(-.05, .05, p.shape)
model.to_gpu()

# Setup optimizer
optimizer = SGD(lr=1.)
optimizer.setup(model.collect_parameters())

# Architecture
def forward_one_step(x_data, y_data, state, train=True):
    x = Variable(x_data, volatile=not train)
    t = Variable(y_data, volatile=not train)
    h0     = model.embed(x)
    h1_in  = model.l1_x(dropout(h0, train=train)) + model.l1_h(state['h1'])
    c1, h1 = lstm(state['c1'], h1_in)
    h2_in  = model.l2_x(dropout(h1, train=train)) + model.l2_h(state['h2'])
    c2, h2 = lstm(state['c2'], h2_in)
    y      = model.l3(dropout(h2, train=train))
    state  = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    return state, softmax_cross_entropy(y, t)

def make_initial_state(batchsize, volatile=False):
    return {name: Variable(gpuarray.zeros((batchsize, n_unit), dtype=np.float32),
                           volatile=volatile)
            for name in ('c1', 'h1', 'c2', 'h2')}

whole_len = train_data.shape[0]
batchsize = 20
bpsize    = 35
grad_clip = 5
showloss  = 10000
n_epoch   = 39
def train_whole():
    jump = whole_len / batchsize
    cur_log_perp = 0
    epoch   = 0

    start_at = time.time()
    cur_at   = start_at
    print 'epoch', epoch

    state = make_initial_state(batchsize=batchsize)
    accum_loss = 0
    for i in xrange(jump * n_epoch):
        x_cpu = np.array([train_data[(jump * j + i) % whole_len]
                          for j in xrange(batchsize)])
        y_cpu = np.array([train_data[(jump * j + i + 1) % whole_len]
                          for j in xrange(batchsize)])
        state, loss_i = forward_one_step(
            gpuarray.to_gpu(x_cpu), gpuarray.to_gpu(y_cpu), state)
        accum_loss   += loss_i
        cur_log_perp += float(loss_i.data.get()) / batchsize

        if (i + 1) % bpsize == 0:
            optimizer.zero_grads()

            accum_loss /= batchsize
            accum_loss.backward()
            accum_loss.forget_backward()
            accum_loss = 0

            optimizer.clip_grads(grad_clip)
            optimizer.update()

        if (i + 1) % showloss == 0:
            now = time.time()
            duration = datetime.timedelta(seconds = now - cur_at)
            print '\nduration:', duration
            cur_at = now

            perp = math.exp(cur_log_perp / showloss)
            print 'iter', i+1, 'training perplexity:', perp
            cur_log_perp = 0

        if (i + 1) % jump == 0:  # epoch
            epoch += 1

            print '\nevaluate'
            perp = evaluate(valid_data)
            print 'epoch', epoch, 'validation perplexity:', perp
            to_eval = False

            print 'epoch', epoch
            if epoch >= 6:
                optimizer.lr /= 1.2
                print 'learning rate =', optimizer.lr

        sys.stderr.write('\rtrain: {} / {}'.format(i + 1, jump*n_epoch))
        sys.stderr.flush()
        sys.stdout.flush()

    print '\ntest'
    test_perp = evaluate(test_data)
    print 'test perplexity:', test_perp
    print '\nwhole time:', datetime.timedelta(seconds = time.time() - start_at)

def evaluate(dataset):
    dataset_len = dataset.size
    sum_log_perp = 0
    state = make_initial_state(batchsize=1, volatile=True)

    start_at = time.time()
    for i in xrange(dataset_len - 1):
        sys.stderr.write('\reval: {} / {}'.format(i, dataset_len - 1))
        sys.stderr.flush()

        x_cpu = dataset[i   : i+1]
        y_cpu = dataset[i+1 : i+2]
        state, loss = forward_one_step(
            gpuarray.to_gpu(x_cpu), gpuarray.to_gpu(y_cpu), state, train=False)
        sum_log_perp += float(loss.data.get())

    print '\nwhole time:', datetime.timedelta(seconds = time.time() - start_at)
    return math.exp(sum_log_perp / (dataset_len - 1))

train_whole()
pickle.dump(model, open('model', 'wb'), -1)
