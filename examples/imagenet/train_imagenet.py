#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle
from six.moves import queue

from chainer import computational_graph as c
from chainer import cuda
from chainer import optimizers

parser = argparse.ArgumentParser(
    description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('train', help='Path to training image-label list file')
parser.add_argument('val', help='Path to validation image-label list file')
parser.add_argument('--mean', '-m', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--arch', '-a', default='nin',
                    help='Convnet architecture \
                    (nin, alexbn, googlenet, googlenetbn)')
parser.add_argument('--batchsize', '-B', type=int, default=32,
                    help='Learning minibatch size')
parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                    help='Validation minibatch size')
parser.add_argument('--epoch', '-E', default=10, type=int,
                    help='Number of epochs to learn')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--loaderjob', '-j', default=20, type=int,
                    help='Number of parallel data loading processes')
parser.add_argument('--out', '-o', default='model',
                    help='Path to save model on each validation')
args = parser.parse_args()
assert 50000 % args.val_batchsize == 0

# Prepare dataset


def load_image_list(path):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append((pair[0], np.int32(pair[1])))
    return tuples

train_list = load_image_list(args.train)
val_list = load_image_list(args.val)
mean_image = pickle.load(open(args.mean, 'rb'))

# Prepare model
if args.arch == 'nin':
    import nin
    model = nin.NIN()
elif args.arch == 'alexbn':
    import alexbn
    model = alexbn.AlexBN()
elif args.arch == 'googlenet':
    import googlenet
    model = googlenet.GoogLeNet()
elif args.arch == 'googlenetbn':
    import googlenetbn
    model = googlenetbn.GoogLeNetBN()
else:
    raise ValueError('Invalid architecture name')

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)


# ------------------------------------------------------------------------------
# This example consists of three threads: data feeder, logger and trainer.
# These communicate with each other via Queue.
data_q = queue.Queue(maxsize=1)
res_q = queue.Queue()

# Data loading routine
cropwidth = 256 - model.insize


def read_image(path, center=False, flip=False):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    if center:
        top = left = cropwidth / 2
    else:
        top = random.randint(0, cropwidth - 1)
        left = random.randint(0, cropwidth - 1)
    bottom = model.insize + top
    right = model.insize + left

    image = image[:, top:bottom, left:right].astype(np.float32)
    image -= mean_image[:, top:bottom, left:right]
    image /= 255
    if flip and random.randint(0, 1) == 0:
        return image[:, :, ::-1]
    else:
        return image

# Data feeder


def feed_data():
    i = 0
    count = 0

    x_batch = np.ndarray(
        (args.batchsize, 3, model.insize, model.insize), dtype=np.float32)
    y_batch = np.ndarray((args.batchsize,), dtype=np.int32)
    val_x_batch = np.ndarray(
        (args.val_batchsize, 3, model.insize, model.insize), dtype=np.float32)
    val_y_batch = np.ndarray((args.val_batchsize,), dtype=np.int32)

    batch_pool = [None] * args.batchsize
    val_batch_pool = [None] * args.val_batchsize
    pool = multiprocessing.Pool(args.loaderjob)
    data_q.put('train')
    for epoch in six.moves.range(1, 1 + args.epoch):
        print('epoch', epoch, file=sys.stderr)
        print('learning rate', optimizer.lr, file=sys.stderr)
        perm = np.random.permutation(len(train_list))
        for idx in perm:
            path, label = train_list[idx]
            batch_pool[i] = pool.apply_async(read_image, (path, False, True))
            y_batch[i] = label
            i += 1

            if i == args.batchsize:
                for j, x in enumerate(batch_pool):
                    x_batch[j] = x.get()
                data_q.put((x_batch.copy(), y_batch.copy()))
                i = 0

            count += 1
            if count % 100000 == 0:
                data_q.put('val')
                j = 0
                for path, label in val_list:
                    val_batch_pool[j] = pool.apply_async(
                        read_image, (path, True, False))
                    val_y_batch[j] = label
                    j += 1

                    if j == args.val_batchsize:
                        for k, x in enumerate(val_batch_pool):
                            val_x_batch[k] = x.get()
                        data_q.put((val_x_batch.copy(), val_y_batch.copy()))
                        j = 0
                data_q.put('train')

        optimizer.lr *= 0.97
    pool.close()
    pool.join()
    data_q.put('end')

# Logger


def log_result():
    train_count = 0
    train_cur_loss = 0
    train_cur_accuracy = 0
    begin_at = time.time()
    val_begin_at = None
    while True:
        result = res_q.get()
        if result == 'end':
            print(file=sys.stderr)
            break
        elif result == 'train':
            print(file=sys.stderr)
            train = True
            if val_begin_at is not None:
                begin_at += time.time() - val_begin_at
                val_begin_at = None
            continue
        elif result == 'val':
            print(file=sys.stderr)
            train = False
            val_count = val_loss = val_accuracy = 0
            val_begin_at = time.time()
            continue

        loss, accuracy = result
        if train:
            train_count += 1
            duration = time.time() - begin_at
            throughput = train_count * args.batchsize / duration
            sys.stderr.write(
                '\rtrain {} updates ({} samples) time: {} ({} images/sec)'
                .format(train_count, train_count * args.batchsize,
                        datetime.timedelta(seconds=duration), throughput))

            train_cur_loss += loss
            train_cur_accuracy += accuracy
            if train_count % 1000 == 0:
                mean_loss = train_cur_loss / 1000
                mean_error = 1 - train_cur_accuracy / 1000
                print(file=sys.stderr)
                print(json.dumps({'type': 'train', 'iteration': train_count,
                                  'error': mean_error, 'loss': mean_loss}))
                sys.stdout.flush()
                train_cur_loss = 0
                train_cur_accuracy = 0
        else:
            val_count += args.val_batchsize
            duration = time.time() - val_begin_at
            throughput = val_count / duration
            sys.stderr.write(
                '\rval   {} batches ({} samples) time: {} ({} images/sec)'
                .format(val_count / args.val_batchsize, val_count,
                        datetime.timedelta(seconds=duration), throughput))

            val_loss += loss
            val_accuracy += accuracy
            if val_count == 50000:
                mean_loss = val_loss * args.val_batchsize / 50000
                mean_error = 1 - val_accuracy * args.val_batchsize / 50000
                print(file=sys.stderr)
                print(json.dumps({'type': 'val', 'iteration': train_count,
                                  'error': mean_error, 'loss': mean_loss}))
                sys.stdout.flush()

# Trainer


def train_loop():
    graph_generated = False
    while True:
        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()
        if inp == 'end':  # quit
            res_q.put('end')
            break
        elif inp == 'train':  # restart training
            res_q.put('train')
            train = True
            continue
        elif inp == 'val':  # start validation
            res_q.put('val')
            pickle.dump(model, open('model', 'wb'), -1)
            train = False
            continue

        x, y = inp
        if args.gpu >= 0:
            x = cuda.to_gpu(x)
            y = cuda.to_gpu(y)

        if train:
            optimizer.zero_grads()
            loss, accuracy = model.forward(x, y)
            loss.backward()
            optimizer.update()

            if not graph_generated:
                with open('graph.dot', 'w') as o:
                    o.write(c.build_computational_graph((loss,), False).dump())
                with open('graph.wo_split.dot', 'w') as o:
                    o.write(c.build_computational_graph((loss,), True).dump())
                print('generated graph')
                graph_generated = True

        else:
            loss, accuracy = model.forward(x, y, train=False)

        res_q.put((float(cuda.to_cpu(loss.data)),
                   float(cuda.to_cpu(accuracy.data))))
        del loss, accuracy, x, y

# Invoke threads
feeder = threading.Thread(target=feed_data)
feeder.daemon = True
feeder.start()
logger = threading.Thread(target=log_result)
logger.daemon = True
logger.start()

train_loop()
feeder.join()
logger.join()

# Save final model
pickle.dump(model, open('model', 'wb'), -1)
