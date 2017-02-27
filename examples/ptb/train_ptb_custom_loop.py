#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

This version implements a custom training loop.
"""
from __future__ import print_function
import argparse
import copy

import chainer
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers

import train_ptb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    args = parser.parse_args()

    def evaluate(model, iter):
        # Evaluation routine to be used for validation and test.
        # This assumes that the mini-batch size is 1.
        model.predictor.train = False
        evaluator = model.copy()  # to use different state
        evaluator.predictor.reset_state()  # initialize state
        evaluator.predictor.train = False  # dropout does nothing
        sum_perp = 0
        data_count = 0
        for batch in copy.copy(iter):
            x, t = convert.concat_examples(batch, args.gpu)
            loss = evaluator(x, t)
            sum_perp += loss.data
            data_count += 1
        model.predictor.train = True
        return sum_perp / data_count

    # Load the Penn Tree Bank long word sequence dataset
    train, val, test = chainer.datasets.get_ptb_words()
    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab =', n_vocab)

    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]

    # Create the dataset iterators
    train_iter = train_ptb.ParallelSequentialIterator(train, args.batchsize)
    val_iter = train_ptb.ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = train_ptb.ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    rnn = train_ptb.RNNForLM(n_vocab, args.unit)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    sum_perp = 0
    count = 0
    iteration = 0
    while train_iter.epoch < args.epoch:
        loss = 0
        iteration += 1
        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(args.bproplen):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()
            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = convert.concat_examples(batch, args.gpu)
            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))
            sum_perp += loss.data
            count += 1

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

        if iteration % 10 == 0:
            print('training perplexity: ', sum_perp / (count*args.batchsize))
            sum_perp = 0
            count = 0

        if train_iter.is_new_epoch:
            perp = evaluate(model, val_iter)
            print('validation perplexity: ', perp)

    # Evaluate on test dataset
    print('test')
    test_perp = evaluate(model, test_iter)
    print('test perplexity:', test_perp)

    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('rnnlm.model', model)
    print('save the optimizer')
    serializers.save_npz('rnnlm.state', optimizer)


if __name__ == '__main__':
    main()