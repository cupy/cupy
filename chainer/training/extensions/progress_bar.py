from __future__ import division
import datetime
import sys
import time

from chainer.training import extension
from chainer.training.extensions import log_report as log_report_module
from chainer.training import trigger


class ProgressBar(extension.Extension):

    """Trainer extension to print a progress bar and recent training status.

    This extension prints a progress bar at every call. It watches the current
    iteration and epoch to print the bar.

    Args:
        training_length (tuple): Length of whole training. It consists of an
            integer and either ``'epoch'`` or ``'iteration'``. If this value is
            omitted and the stop trigger of the trainer is
            :class:`IntervalTrigger`, this extension uses its attributes to
            determine the length of the training.
        iterations_per_epoch (int): Number of iterations per epoch. This value
            is used to print the bar when the training length is specified by
            a number of epochs.
        update_interval (int): Number of iterations to skip printing the
            progress bar.
        bar_length (int): Length of the progress bar.
        out: Stream to print the bar. Standard output is used by default.

    """
    def __init__(self, training_length=None, iterations_per_epoch=None,
                 update_interval=100, bar_length=50, out=sys.stdout):
        self._training_length = training_length
        self._status_template = None
        self._iterations_per_epoch = iterations_per_epoch
        self._update_interval = update_interval
        self._bar_length = bar_length
        self._out = out
        self._recent_timing = []

    def __call__(self, trainer):
        training_length = self._training_length

        # initialize some attributes at the first call
        if training_length is None:
            t = trainer.stop_trigger
            if not isinstance(t, trigger.IntervalTrigger):
                raise TypeError('cannot retrieve the training length from'
                                '%s' % type(stop_trigger))
            training_length = self._training_length = t.period, t.unit

        stat_template = self._status_template
        if stat_template is None:
            stat_template = self._status_template = (
                '{0.iteration} iter, {0.epoch} epoch / %s %ss' %
                training_length)

        iters, unit = training_length
        if unit == 'epoch':
            iter_per_epoch = self._iterations_per_epoch
            if iter_per_epoch is None:
                raise ValueError('needs iterations_per_epoch set when the '
                                 'training length is given in epochs')
            iters *= iter_per_epoch

        out = self._out

        # print the progress bar
        iteration = trainer.updater.iteration
        if iteration % self._update_interval == 0:
            recent_timing = self._recent_timing
            now = time.clock()

            if len(recent_timing) >= 1:
                out.write('\x1b\x9bJ')

                bar_length = self._bar_length
                rate = iteration / iters
                marks = '#' * int(rate * bar_length)
                bar = '[{}{}] {:.4%}\n'.format(
                    marks, '.' * (bar_length - len(marks)), rate)
                out.write(bar)

                status = stat_template.format(trainer.updater)
                old_t, old_sec = recent_timing[0]
                speed = (iteration - old_t) / (now - old_sec)
                estimated_time = (iters - iteration) / speed
                out.write('{:.5g} iters/sec.\tEstimated time to finish: {}.\n'
                          .format(speed,
                                  datetime.timedelta(seconds=estimated_time)))

                # move the cursor to the head of the progress bar
                out.write('\x1b\x9b2A')
                out.flush()

                if len(recent_timing) > 100:
                    del recent_timing[0]

            recent_timing.append((iteration, now))
