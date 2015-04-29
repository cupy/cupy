#!/usr/bin/env python
import argparse, json, math, cPickle as pickle, Queue, random, sys, threading, time
from datetime import timedelta
import cv2, numpy as np
from chainer      import cuda, Variable, FunctionSet
from chainer.cuda import to_gpu
import chainer.functions as F
import chainer.optimizers as O

from inception import InceptionBN

parser = argparse.ArgumentParser(description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('--arch', '-a', default='inceptionbn',
                    help='convnet architecture (nin, inceptionbn, alexbn)')
parser.add_argument('--batchsize', '-B', type=int, default=32,
                    help='learning batchsize')
parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                    help='validation batchsize (must be divide 50000)')
args = parser.parse_args()

batchsize = args.batchsize
val_batchsize = args.val_batchsize
assert 50000 % val_batchsize == 0

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

if args.arch == 'inceptionbn':
    insize = 224
elif args.arch == 'nin' or args.arch == 'alexbn':
    insize = 227
cropwidth = 256 - insize

def read_image(path, center=False, flip=False):
    image = cv2.imread(path)
    if center:
        top = left = cropwidth / 2
    else:
        top  = random.randint(0, cropwidth - 1)
        left = random.randint(0, cropwidth - 1)
    bottom = 256 - (cropwidth - top)
    right  = 256 - (cropwidth - left)

    image = image[top:bottom, left:right, [2, 1, 0]].transpose(2, 0, 1).astype(np.float32)
    image -= mean_image[:, top:bottom, left:right]
    image /= 255
    if flip and random.randint(0, 1) == 0:
        image = image[:, :, ::-1]

    return image

# Inter-thread communication
data_q = Queue.Queue(maxsize=1)
res_q  = Queue.Queue()

# data feeder
n_epoch = 6
x_batch = np.ndarray((batchsize, 3, insize, insize), dtype=np.float32)
y_batch = np.ndarray((batchsize,), dtype=np.int32)
val_x_batch = np.ndarray((val_batchsize, 3, insize, insize), dtype=np.float32)
val_y_batch = np.ndarray((val_batchsize,), dtype=np.int32)
def feed_data():
    i = 0
    count = 0
    for epoch in xrange(1, 1 + n_epoch):
        print >> sys.stderr, 'epoch', epoch
        print >> sys.stderr, 'learning rate', optimizer.lr
        perm = np.random.permutation(len(train_list))
        if i == 0:
            data_q.put('train')
        for idx in perm:
            path, label = train_list[idx]
            x_batch[i] = read_image(path, flip=True)
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
    begin_at = time.time()
    val_begin_at = None
    while True:
        result = res_q.get()
        if result == 'end':
            print >> sys.stderr, ''
            break
        elif result == 'train':
            print >> sys.stderr, ''
            train = True
            if val_begin_at is not None:
                begin_at += time.time() - val_begin_at
                val_begin_at = None
            continue
        elif result == 'val':
            print >> sys.stderr, ''
            train = False
            val_count = val_loss = val_accuracy = 0
            val_begin_at = time.time()
            continue

        loss, accuracy = result
        if train:
            train_count += 1
            duration = time.time() - begin_at
            t_per_sample = duration / (train_count * batchsize)
            sys.stderr.write(
                '\rtrain {} updates ({} samples) time: {} ({} sec/sample)'
                .format(train_count, train_count * batchsize,
                        timedelta(seconds=duration), t_per_sample))

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
            duration = time.time() - val_begin_at
            t_per_samle = duration / val_count
            sys.stderr.write(
                '\rval   {} batches ({} samples) time: {} ({} sec/sample)'
                .format(val_count / val_batchsize, val_count,
                        timedelta(seconds=duration), t_per_samle))

            val_loss += loss
            val_accuracy += accuracy
            if val_count == 50000:
                mean_loss  = val_loss * val_batchsize / 50000
                mean_error = 1 - val_accuracy * val_batchsize / 50000
                print >> sys.stderr, ''
                print json.dumps({'type': 'val', 'iteration': train_count,
                                  'error': mean_error, 'loss': mean_loss})
                sys.stdout.flush()

cuda.init()

# model
if args.arch == 'inceptionbn':
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

        conva  = F.Convolution2D(576, 128, 1, nobias=True),
        norma  = F.BatchNormalization(128),
        lina   = F.Linear(2048, 1024, nobias=True),
        norma2 = F.BatchNormalization(1024),
        outa   = F.Linear(1024, 1000),

        convb  = F.Convolution2D(576, 128, 1, nobias=True),
        normb  = F.BatchNormalization(128),
        linb   = F.Linear(2048, 1024, nobias=True),
        normb2 = F.BatchNormalization(1024),
        outb   = F.Linear(1024, 1000),
    )
elif args.arch == 'nin':
    w = math.sqrt(2)  # MSRA scaling
    model = FunctionSet(
        conv1  = F.Convolution2D(   3,   96, 11, wscale=w, stride=4),
        conv1a = F.Convolution2D(  96,   96,  1, wscale=w),
        conv1b = F.Convolution2D(  96,   96,  1, wscale=w),
        conv2  = F.Convolution2D(  96,  256,  5, wscale=w, pad=2),
        conv2a = F.Convolution2D( 256,  256,  1, wscale=w),
        conv2b = F.Convolution2D( 256,  256,  1, wscale=w),
        conv3  = F.Convolution2D( 256,  384,  3, wscale=w, pad=1),
        conv3a = F.Convolution2D( 384,  384,  1, wscale=w),
        conv3b = F.Convolution2D( 384,  384,  1, wscale=w),
        conv4  = F.Convolution2D( 384, 1024,  3, wscale=w, pad=1),
        conv4a = F.Convolution2D(1024, 1024,  1, wscale=w),
        conv4b = F.Convolution2D(1024, 1000,  1, wscale=w),
    )
elif args.arch == 'alexbn':
    model = FunctionSet(
        conv1 = F.Convolution2D(  3,  96, 11, stride=4),
        bn1   = F.BatchNormalization( 96),
        conv2 = F.Convolution2D( 96, 256,  5, pad=2),
        bn2   = F.BatchNormalization(256),
        conv3 = F.Convolution2D(256, 384,  3, pad=1),
        conv4 = F.Convolution2D(384, 384,  3, pad=1),
        conv5 = F.Convolution2D(384, 256,  3, pad=1),
        fc6   = F.Linear(9216, 4096),
        fc7   = F.Linear(4096, 4096),
        fc8   = F.Linear(4096, 1000),
    )
else:
    raise NotImplementedError()

model.to_gpu()

# Architecture
def forward(x_data, y_data, volatile=False):
    x = Variable(x_data, volatile=volatile)
    t = Variable(y_data, volatile=volatile)

    if args.arch == 'inceptionbn':
        h = F.max_pooling_2d(F.relu(model.norm1(model.conv1(x))),  3, stride=2, pad=1)
        h = F.max_pooling_2d(F.relu(model.norm2(model.conv2(h))), 3, stride=2, pad=1)

        h = model.inc3a(h)
        h = model.inc3b(h)
        h = model.inc3c(h)
        h = model.inc4a(h)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = F.relu(model.norma(model.conva(a)))
        a = F.relu(model.norma2(model.lina(a)))
        a = model.outa(a)
        a = F.softmax_cross_entropy(a, t)

        h = model.inc4b(h)
        h = model.inc4c(h)
        h = model.inc4d(h)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = F.relu(model.normb(model.convb(b)))
        b = F.relu(model.normb2(model.linb(b)))
        b = model.outb(b)
        b = F.softmax_cross_entropy(b, t)

        h = model.inc4e(h)
        h = model.inc5a(h)
        h = F.average_pooling_2d(model.inc5b(h), 7)
        h = model.out(h)
        l = F.softmax_cross_entropy(h, t)
        acc = F.accuracy(h, t)
        L   = (a * 0.3 + b * 0.3 + l) / x_data.shape[0]
    elif args.arch == 'nin':
        h = F.relu(model.conv1(x))
        h = F.relu(model.conv1a(h))
        h = F.relu(model.conv1b(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(model.conv2(h))
        h = F.relu(model.conv2a(h))
        h = F.relu(model.conv2b(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(model.conv3(h))
        h = F.relu(model.conv3a(h))
        h = F.relu(model.conv3b(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.dropout(h, train=not volatile)
        h = F.relu(model.conv4(h))
        h = F.relu(model.conv4a(h))
        h = F.relu(model.conv4b(h))
        h = F.average_pooling_2d(h, 6)
        l = F.softmax_cross_entropy(h, t)
        acc = F.accuracy(h, t)
        L = l / x_data.shape[0]
    elif args.arch == 'alexbn':
        h = F.max_pooling_2d(F.relu(model.bn1(model.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(model.bn2(model.conv2(h))), 3, stride=2)
        h = F.relu(model.conv3(h))
        h = F.relu(model.conv4(h))
        h = F.max_pooling_2d(F.relu(model.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(model.fc6(h)))
        h = F.dropout(F.relu(model.fc7(h)))
        h = model.fc8(h)
        l = F.softmax_cross_entropy(h, t)
        acc = F.accuracy(h, t)
        L = l / x_data.shape[0]

    return L, acc

# Setup optimizer
optimizer = O.MomentumSGD(lr=0.01, momentum=0.9)
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
