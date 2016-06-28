#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

import train_mnist


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=400,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu0', '-g', type=int, default=0,
                        help='First GPU ID')
    parser.add_argument('--gpu1', '-G', type=int, default=1,
                        help='Second GPU ID')
    parser.add_argument('--out', '-o', default='result_parallel',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}, {}'.format(args.gpu0, args.gpu1))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # See train_mnist.py for the meaning of these lines
    model_0 = L.Classifier(train_mnist.MLP(784, args.unit, 10))
    model_1 = model_0.copy()
    model_0.to_gpu(args.gpu0)
    model_1.to_gpu(args.gpu1)

    # Make a specified GPU current
    chainer.cuda.get_device(args.gpu0).use()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model_0)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.ShuffledIterator(train, args.batchsize)
    test_iter = chainer.iterators.SequentialIterator(test, args.batchsize,
                                                     repeat=False)

    # Set up a trainer
    updater = training.ParallelUpdater(
        train_iter,
        optimizer,
        models={'main':model_0, 'second':model_1},
        devices={'main':args.gpu0, 'second':args.gpu1},
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Add observer for model on second gpu.
    trainer.reporter.add_observer('second', model_1)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model_0, device=args.gpu0))

    # Dump a computational graph from 'loss' variable at the first iteration
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot())

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Entries prefixed by 'validation' are computed by the Evaluator extension
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
