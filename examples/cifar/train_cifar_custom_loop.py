from __future__ import print_function
import argparse
import copy

import chainer
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import models.VGG


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')

    if args.test:
        train = train[:200]
        test = test[:200]

    train_count = len(train)
    test_count = len(test)

    model = L.Classifier(models.VGG.VGG(class_labels))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(0.1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    sum_accuracy = 0
    sum_loss = 0

    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        # Reduce learning rate by 0.5 every 25 epochs.
        if train_iter.epoch % 25 == 0 and train_iter.is_new_epoch:
            optimizer.lr *= 0.5
            print('Reducing learning rate to: ', optimizer.lr)

        x_array, t_array = convert.concat_examples(batch, args.gpu)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

        if train_iter.is_new_epoch:
            print('train mean loss={}, accuracy={}'.format(
                sum_loss / train_count, sum_accuracy / train_count))
            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            model.predictor.train = False
            for batch in copy.copy(test_iter):
                x_array, t_array = convert.concat_examples(batch, args.gpu)
                x = chainer.Variable(x_array)
                t = chainer.Variable(t_array)
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)

            model.predictor.train = True
            print('test mean loss={}, accuracy={}'.format(
                sum_loss / test_count, sum_accuracy / test_count))
            sum_accuracy = 0
            sum_loss = 0

    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('mlp.model', model)
    print('save the optimizer')
    serializers.save_npz('mlp.state', optimizer)

if __name__ == '__main__':
    main()
