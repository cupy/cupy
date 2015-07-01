#!/usr/bin/env python
"""Example code of evaluating a Caffe reference model for ILSVRC2012 task.

Prerequisite: To run this example, crop the center of ILSVRC2012 validation
images and scale them to 256x256, and make a list of space-separated CSV each
column of which contains a full path to an image at the fist column and a zero-
origin label at the second column (this format is same as that used by Caffe's
ImageDataLayer).

"""
from __future__ import print_function
import argparse
import multiprocessing
import sys
import threading
import time

import cv2
import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe


parser = argparse.ArgumentParser(
    description='Evaluate a Caffe reference model on ILSVRC2012 dataset')
parser.add_argument('dataset', help='Path to validation image-label list file')
parser.add_argument('model_type',
                    help='Model type (alexnet, caffenet, googlenet)')
parser.add_argument('model', help='Path to the pretrained Caffe model')
parser.add_argument('--mean', '-m', default='ilsvrc_2012_mean.npy',
                    help='Path to the mean file')
parser.add_argument('--batchsize', '-B', type=int, default=100,
                    help='Minibatch size')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='Zero-origin GPU ID (nevative value indicates CPU)')
parser.add_argument('--loaderjob', '-j', default=20, type=int,
                    help='Number of parallel data loading processes')
args = parser.parse_args()


def load_image_list(path):
    tuples = []
    with open(path) as list_file:
        for line in list_file:
            pair = line.strip().split()
            tuples.append((pair[0], np.int32(pair[1])))
    return tuples


dataset = load_image_list(args.dataset)
assert len(dataset) % args.batchsize == 0


# Prepare the model type
if args.model_type == 'alexnet' or args.model_type == 'caffenet':
    in_size = 227
    data_name = 'data'
    out_name = 'fc8'
    ignores = []
    mean_image = np.load(args.mean)
elif args.model_type == 'googlenet':
    in_size = 224
    data_name = 'data'
    out_name = 'loss3/classifier'
    ignores = ['loss1/ave_pool', 'loss2/ave_pool']
    # Constant mean over spatial pixels
    mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
    mean_image[0] = 104
    mean_image[1] = 117
    mean_image[2] = 123
else:
    print('Invalid model type name. Choose from alexnet, caffenet, googlenet.')
    exit(1)


cropwidth = 256 - in_size
start = cropwidth // 2
stop = start + in_size
mean_image = mean_image[:, start:stop, start:stop].copy()


func = caffe.CaffeFunction(args.model)
if args.gpu >= 0:
    cuda.init(args.gpu)
    func.to_gpu()


# -----------------------------------------------------------------------------
# This example consists of three threads: data feeder, logger and evaluator.
# These communicate with each other via Queue.
data_q = six.moves.queue.Queue(maxsize=1)
res_q = six.moves.queue.Queue()


def read_image(path):
    # Data loading routine
    # TODO(beam2d): Remove dependency on OpenCV
    image = cv2.imread(path).transpose(2, 0, 1)
    image = image[[2, 1, 0], start:stop, start:stop].astype(np.float32)
    return image - mean_image


def feed_data():
    # Data feeder
    x_batch = np.ndarray(
        (args.batchsize, 3, in_size, in_size), dtype=np.float32)
    y_batch = np.ndarray((args.batchsize,), dtype=np.int32)

    batch_pool = [None] * args.batchsize
    pool = multiprocessing.Pool(args.loaderjob)

    i = 0
    for path, label in dataset:
        batch_pool[i] = pool.apply_async(read_image, (path,))
        y_batch[i] = label
        i += 1

        if i == args.batchsize:
            for j, x in enumerate(batch_pool):
                x_batch[j] = x.get()
            data_q.put((x_batch.copy(), y_batch.copy()))
            i = 0

    pool.close()
    pool.join()
    data_q.put('end')


def log_result():
    # Logger
    count = 0
    val_loss = 0
    val_accuracy = 0
    while True:
        result = res_q.get()
        if result == 'end':
            print(file=sys.stderr)
            break

        loss, accuracy = result
        count += args.batchsize
        val_loss += loss * args.batchsize
        val_accuracy += accuracy * args.batchsize

        print('{} / {}'.format(count, len(dataset)), end='\r', file=sys.stderr)
        sys.stderr.flush()

    print('mean loss:     {}'.format(val_loss / count))
    print('mean accuracy: {}'.format(val_accuracy / count))


def eval_loop():
    # Evaluator
    while True:
        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()
        if inp == 'end':
            res_q.put('end')
            break

        x_data, y_data = inp
        if args.gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)

        x = chainer.Variable(x_data, volatile=True)
        t = chainer.Variable(y_data, volatile=True)

        y, = func(inputs={data_name: x}, outputs=[out_name], disable=ignores)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)

        res_q.put((float(cuda.to_cpu(loss.data)),
                   float(cuda.to_cpu(accuracy.data))))

        del loss, accuracy, x, t, y


# Invoke threads
feeder = threading.Thread(target=feed_data)
feeder.daemon = True
feeder.start()
logger = threading.Thread(target=log_result)
logger.daemon = True
logger.start()

eval_loop()
feeder.join()
logger.join()
