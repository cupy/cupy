#!/usr/bin/env python
import json, cPickle as pickle, Queue, random, sys, threading
import cv2, numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.gpuarray import to_gpu
from chainer import Variable, FunctionSet
import chainer.functions as F
import chainer.optimizers as O

from inception import InceptionBN

def indicate(i, n):
    sys.stderr.write('\r{} / {}'.format(i, n))
    sys.stderr.flush()

def load_image_list(path):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append((pair[0], np.int32(pair[1])))
    return tuples

train_path = '/data/ILSVRC/train.txt'
val_path   = '/data/ILSVRC/val.txt'
train_list = load_image_list(train_path)
val_list   = load_image_list(val_path)

mean_image = pickle.load(open('mean.npy', 'rb'))

# model
model = FunctionSet(
    conv1 = F.Convolution2D( 3,  64, 7, stride=2, pad=3),
    norm1 = F.BatchNormalization(64),
    conv2 = F.Convolution2D(64, 192, 3, pad=1),
    norm2 = F.BatchNormalization(192),
    inc3a = InceptionBN( 192,  64,  64,  64,  64,  96, 'avg',  32),
    inc3b = InceptionBN( 256,  64,  64,  96,  64,  96, 'avg',  64),
    inc3c = InceptionBN( 320,   0, 128, 160,  64,  96, 'max', stride=2),
    inc4a = InceptionBN( 576, 224,  64,  96,  96, 128, 'avg', 128),
    inc4b = InceptionBN( 576, 192,  96, 128,  96, 128, 'avg', 128),
    inc4c = InceptionBN( 576, 128, 128, 160, 128, 160, 'avg', 128),
    inc4d = InceptionBN( 576,  64, 128, 192, 160, 192, 'avg', 128),
    inc4e = InceptionBN( 576,   0, 128, 192, 192, 256, 'max', stride=2),
    inc5a = InceptionBN(1024, 352, 192, 320, 160, 224, 'avg', 128),
    inc5b = InceptionBN(1024, 352, 192, 320, 192, 224, 'max', 128),
    out   = F.Linear(1024, 1024),
)

# Architecture
def forward(x_data, y_data, volatile=False):
    x = Variable(x_data, volatile=volatile)
    t = Variable(y_data, volatile=volatile)
    h1 = F.max_pooling_2d(F.relu(model.norm1(model.conv1(x))),  3, stride=2, pad=1)
    h2 = F.max_pooling_2d(F.relu(model.norm2(model.conv2(h1))), 3, stride=2, pad=1)
    h5  = model.inc3a(h2)
    h8  = model.inc3b(h5)
    h11 = model.inc3c(h8)
    h14 = model.inc4a(h11)
    h17 = model.inc4b(h14)
    h20 = model.inc4c(h17)
    h23 = model.inc4d(h20)
    h26 = model.inc4e(h23)
    h29 = model.inc5a(h26)
    h32 = F.average_pooling_2d(model.inc5b(h29), 7)
    y   = model.out(h32)
    return F.softmax_cross_entropy(y, t) / x_data.shape[0], F.accuracy(y, t)

# Setup optimizer
optimizer = O.MomentumSGD(lr=0.045, momentum=0.9)

# Learning loop
trash_q  = Queue.Queue()
data_q   = Queue.Queue(maxsize=1)
result_q = Queue.Queue()
N     = len(train_list)
N_val = len(val_list)
def train_loop():
    device = pycuda.autoinit.device
    ctx = device.make_context()
    # GPU migration must be taken here with the context created on this thread
    model.to_gpu()
    optimizer.setup(model.collect_parameters())

    try:
        while True:
            batch = data_q.get()
            if batch == 'end':  # quit
                data_q.task_done()
                result_q.put('end')
                break
            elif batch == 'train':  # restart training
                data_q.task_done()
                result_q.put('train')
                train = True
                continue
            elif batch == 'val':  # start validation
                data_q.task_done()
                result_q.put('val')
                train = False
                continue

            if train:
                loss, accuracy = forward(*batch)
                loss.backward()
                optimizer.update()
            else:
                loss, accuracy = forward(*batch, volatile=True)

            result_q.put((float(loss.data.get()), float(accuracy.data.get())))
            del loss, accuracy

            trash_q.put(batch)
            del batch

            data_q.task_done()
    finally:
        cuda.Context.pop()

batchsize = 32
val_batchsize = 256

# Show result
def show_result():
    train_count = 0
    train_cur_loss = 0
    train_cur_accuracy = 0
    while True:
        result = result_q.get()
        if result == 'end':
            print >> sys.stderr, ''
            result_q.task_done()
            break
        elif result == 'train':
            print >> sys.stderr, ''
            train = True
            result_q.task_done()
            continue
        elif result == 'val':
            print >> sys.stderr, ''
            train = False
            val_count = val_loss = val_accuracy = 0
            result_q.task_done()
            continue

        loss, accuracy = result
        if train:
            train_count += 1
            sys.stderr.write('\rtrain {} updates ({} samples)'.format(train_count, train_count * batchsize))

            train_cur_loss += loss
            train_cur_accuracy += accuracy
            if train_count % 1000 == 0:
                mean_loss  = train_cur_loss / 1000
                mean_error = 1 - train_cur_accuracy / 1000
                print json.dumps({'type': 'train', 'iteration': train_count,
                                  'error': mean_error, 'loss': mean_loss})
                train_cur_loss = 0
                train_cur_accuracy = 0
        else:
            val_count += val_batchsize
            sys.stderr.write('\rval   {} batches ({} samples)'.format(val_count / val_batchsize, val_count))

            val_loss += loss
            val_accuracy += accuracy
            if val_count == 50000:
                mean_loss  = val_loss * val_batchsize / 50000
                mean_error = 1 - val_accuarcy * val_batchsize / 50000
                print json.dumps({'type': 'val', 'iteration': train_count,
                                  'error': mean_error, 'loss': mean_loss})

        result_q.task_done()

def read_image(path, center=False):
    image = cv2.imread(path)
    if center:
        top = left = 16
    else:
        top  = random.randint(0, 31)
        left = random.randint(0, 31)
    bottom = 256 - (32 - top)
    right  = 256 - (32 - left)

    image = image[top:bottom, left:right, [2, 1, 0]].transpose(2, 0, 1).astype(np.float32)
    image -= mean_image[:, top:bottom, left:right]
    image /= 255
    return image

def trash():
    while True:
        try:
            trash_q.get_nowait()
            trash_q.task_done()
        except:
            break

# data feeder
x_batch = np.ndarray((batchsize, 3, 224, 224), dtype=np.float32)
y_batch = np.ndarray((batchsize,), dtype=np.int32)
val_x_batch = np.ndarray((val_batchsize, 3, 224, 224), dtype=np.float32)
val_y_batch = np.ndarray((val_batchsize,), dtype=np.int32)
def feed_data():
    perm = np.random.permutation(len(train_list))
    data_q.put('train')
    i = 0
    count = 0
    for idx in perm:
        path, label = train_list[idx]
        x_batch[i] = read_image(path)
        y_batch[i] = label
        i += 1

        if i == batchsize:
            data_q.put((to_gpu(x_batch), to_gpu(y_batch)))
            i = 0

        count += 1
        if count % 10000 == 0:
            data_q.join()
            trash()
            data_q.put('val')
            j = 0
            for path, label in val_list:
                val_x_batch[j] = read_image(path, center=True)
                val_y_batch[j] = label
                j += 1

                if j == val_batchsize:
                    data_q.put((to_gpu(val_x_batch), to_gpu(val_y_batch)))
                    j = 0
                trash()
        trash()

learner = threading.Thread(target=train_loop)
learner.start()
shower  = threading.Thread(target=show_result)
shower.start()

for epoch in xrange(5):
    print >> sys.stderr, 'epoch', epoch
    print >> sys.stderr, 'learning rate', optimizer.lr
    feed_data()
    optimizer.lr *= 0.97
