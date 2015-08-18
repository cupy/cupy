#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST using two GPUs.

This is a toy example to write a model-parallel computation in Chainer.
Note that this is just an example; the network definition is not optimal
and performs poorly on MNIST dataset.

"""
import math

import numpy as np
import six

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import optimizers

import data

batchsize = 100
n_epoch = 50
n_units = 2000

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

# Prepare the multi-layer perceptron model
# Note that the model splits into two GPUs at the first layer,
# and share their activations only at third and sixth layers.
cuda.init()
wscale = math.sqrt(2)
model = chainer.FunctionSet(
    gpu0=chainer.FunctionSet(
        l1=F.Linear(784, n_units // 2, wscale=wscale),
        l2=F.Linear(n_units // 2, n_units // 2, wscale=wscale),
        l3=F.Linear(n_units // 2, n_units,      wscale=wscale),
        l4=F.Linear(n_units,      n_units // 2, wscale=wscale),
        l5=F.Linear(n_units // 2, n_units // 2, wscale=wscale),
        l6=F.Linear(n_units // 2, 10,           wscale=wscale)
    ).to_gpu(0),
    gpu1=chainer.FunctionSet(
        l1=F.Linear(784, n_units // 2, wscale=wscale),
        l2=F.Linear(n_units // 2, n_units // 2, wscale=wscale),
        l3=F.Linear(n_units // 2, n_units,      wscale=wscale),
        l4=F.Linear(n_units,      n_units // 2, wscale=wscale),
        l5=F.Linear(n_units // 2, n_units // 2, wscale=wscale),
        l6=F.Linear(n_units // 2, 10,           wscale=wscale)
    ).to_gpu(1)
)
optimizer = optimizers.SGD(lr=0.1)
optimizer.setup(model)

# Neural net architecture


def forward(x_data, y_data, train=True):
    x_0 = chainer.Variable(cuda.to_gpu(x_data, 0), volatile=not train)
    x_1 = chainer.Variable(cuda.to_gpu(x_data, 1), volatile=not train)
    t = chainer.Variable(cuda.to_gpu(y_data, 0), volatile=not train)

    h1_0 = F.dropout(F.relu(model.gpu0.l1(x_0)),  train=train)
    h1_1 = F.dropout(F.relu(model.gpu1.l1(x_1)),  train=train)

    h2_0 = F.dropout(F.relu(model.gpu0.l2(h1_0)), train=train)
    h2_1 = F.dropout(F.relu(model.gpu1.l2(h1_1)), train=train)

    h3_0 = F.dropout(F.relu(model.gpu0.l3(h2_0)), train=train)
    h3_1 = F.dropout(F.relu(model.gpu1.l3(h2_1)), train=train)

    # Synchronize
    h3_0 += F.copy(h3_1, 0)
    h3_1 = F.copy(h3_0, 1)

    h4_0 = F.dropout(F.relu(model.gpu0.l4(h3_0)), train=train)
    h4_1 = F.dropout(F.relu(model.gpu1.l4(h3_1)), train=train)

    h5_0 = F.dropout(F.relu(model.gpu0.l5(h4_0)),  train=train)
    h5_1 = F.dropout(F.relu(model.gpu1.l5(h4_1)),  train=train)

    h6_0 = F.relu(model.gpu0.l6(h5_0))
    h6_1 = F.relu(model.gpu1.l6(h5_1))

    # Synchronize
    y = h6_0 + F.copy(h6_1, 0)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# Learning loop
x_batch = np.ndarray((batchsize, 784), dtype=np.float32)
y_batch = np.ndarray((batchsize,), dtype=np.int32)
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch[:] = x_train[perm[i:i + batchsize]]
        y_batch[:] = y_train[perm[i:i + batchsize]]

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        loss, acc = forward(x_test[i:i + batchsize], y_test[i:i + batchsize],
                            train=False)

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
