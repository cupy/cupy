#!/usr/bin/env python
import json, cPickle as pickle, Queue, random, sys, threading
import cv2, numpy as np
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

batchsize = 32
val_batchsize = 250

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

# Inter-thread communication
data_q = Queue.Queue(maxsize=1)
res_q  = Queue.Queue()

# data feeder
n_epoch = 6
x_batch = np.ndarray((batchsize, 3, 224, 224), dtype=np.float32)
y_batch = np.ndarray((batchsize,), dtype=np.int32)
val_x_batch = np.ndarray((val_batchsize, 3, 224, 224), dtype=np.float32)
val_y_batch = np.ndarray((val_batchsize,), dtype=np.int32)
def feed_data():
    for epoch in xrange(1, 1 + n_epoch):
        print >> sys.stderr, 'epoch', epoch
        print >> sys.stderr, 'learning rate', optimizer.lr
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
                data_q.put((x_batch, y_batch))
                i = 0

            count += 1
            if count % 100000 == 0:
                data_q.put('val')
                j = 0
                for path, label in val_list:
                    val_x_batch[j] = read_image(path, center=True)
                    val_y_batch[j] = label
                    j += 1

                    if j == val_batchsize:
                        data_q.put((val_x_batch, val_y_batch))
                        j = 0
                data_q.put('train')

        optimizer.lr *= 0.97

# Log result
def log_result():
    train_count = 0
    train_cur_loss = 0
    train_cur_accuracy = 0
    while True:
        result = res_q.get()
        if result == 'end':
            print >> sys.stderr, ''
            break
        elif result == 'train':
            print >> sys.stderr, ''
            train = True
            continue
        elif result == 'val':
            print >> sys.stderr, ''
            train = False
            val_count = val_loss = val_accuracy = 0
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
                print >> sys.stderr, ''
                print json.dumps({'type': 'train', 'iteration': train_count,
                                  'error': mean_error, 'loss': mean_loss})
                sys.stdout.flush()
                train_cur_loss = 0
                train_cur_accuracy = 0
        else:
            val_count += val_batchsize
            sys.stderr.write('\rval   {} batches ({} samples)'.format(val_count / val_batchsize, val_count))

            val_loss += loss
            val_accuracy += accuracy
            if val_count == 50000:
                mean_loss  = val_loss * val_batchsize / 50000
                mean_error = 1 - val_accuracy * val_batchsize / 50000
                print >> sys.stderr, ''
                print json.dumps({'type': 'val', 'iteration': train_count,
                                  'error': mean_error, 'loss': mean_loss})
                sys.stdout.flush()

# model
model = FunctionSet(
    conv1 = F.Convolution2D( 3,  64, 7, stride=2, pad=3, nobias=True),
    norm1 = F.BatchNormalization(64),
    conv2 = F.Convolution2D(64, 192, 3, pad=1, nobias=True),
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
    out   = F.Linear(1024, 1000),

    conva  = F.Convolution2D(576, 64, 1, nobias=True),
    norma  = F.BatchNormalization(64),
    lina   = F.Linear(1024, 1024, nobias=True),
    norma2 = F.BatchNormalization(1024),
    outa   = F.Linear(1024, 1000),

    convb  = F.Convolution2D(576, 64, 1, nobias=True),
    normb  = F.BatchNormalization(64),
    linb   = F.Linear(1024, 1024, nobias=True),
    normb2 = F.BatchNormalization(1024),
    outb   = F.Linear(1024, 1000),
)
model.to_gpu()

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

    a14 = F.average_pooling_2d(h14, 5, stride=3)
    a15 = F.relu(model.norma(model.conva(a14)))
    a16 = F.relu(model.norma2(model.lina(a15)))
    ya  = model.outa(a16)
    la  = F.softmax_cross_entropy(ya, t)

    h17 = model.inc4b(h14)
    h20 = model.inc4c(h17)
    h23 = model.inc4d(h20)

    b23 = F.average_pooling_2d(h23, 5, stride=3)
    b24 = F.relu(model.normb(model.convb(b23)))
    b25 = F.relu(model.normb2(model.linb(b24)))
    yb  = model.outb(b25)
    lb  = F.softmax_cross_entropy(yb, t)

    h26 = model.inc4e(h23)
    h29 = model.inc5a(h26)
    h32 = F.average_pooling_2d(model.inc5b(h29), 7)
    y   = model.out(h32)
    l   = F.softmax_cross_entropy(y, t)

    acc = F.accuracy(y, t)

    L   = (la * 0.3 + lb * 0.3 + l) / x_data.shape[0]
    return L, acc

# Setup optimizer
optimizer = O.MomentumSGD(lr=0.0075, momentum=0.9)
optimizer.setup(model.collect_parameters())

# Main loop
def train_loop():
    while True:
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
            train = False
            continue

        x = to_gpu(inp[0])
        y = to_gpu(inp[1])

        if train:
            optimizer.zero_grads()
            loss, accuracy = forward(x, y)
            loss.backward()
            optimizer.update()
        else:
            loss, accuracy = forward(x, y, volatile=True)

        res_q.put((float(loss.data.get()), float(accuracy.data.get())))
        del loss, accuracy
        del x, y

feeder = threading.Thread(target=feed_data)
feeder.start()
logger = threading.Thread(target=log_result)
logger.start()

train_loop()
feeder.join()
logger.join()

pickle.dump(model, open('model', 'wb'), -1)
