from __future__ import division
import datetime
import os
import sys
import time

from chainer.training import extension
from chainer.training.extensions import util
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
        update_interval (int): Number of iterations to skip printing the
            progress bar.
        bar_length (int): Length of the progress bar in characters.
        out: Stream to print the bar. Standard output is used by default.

    """

    def __init__(self, training_length=None, update_interval=100,
                 bar_length=50, out=sys.stdout):
        self._training_length = training_length
        self._status_template = None
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
                raise TypeError(
                    'cannot retrieve the training length from %s' % type(t))
            training_length = self._training_length = t.period, t.unit

        stat_template = self._status_template
        if stat_template is None:
            stat_template = self._status_template = (
                '{0.iteration:10} iter, {0.epoch} epoch / %s %ss\n' %
                training_length)

        length, unit = training_length
        out = self._out

        iteration = trainer.updater.iteration

        # print the progress bar
        if iteration % self._update_interval == 0:
            epoch = trainer.updater.epoch_detail
            recent_timing = self._recent_timing
            now = time.time()

            recent_timing.append((iteration, epoch, now))

            if os.name == 'nt':
                util.erase_console(0, 0)
            else:
                out.write('\033[J')

            if unit == 'iteration':
                rate = iteration / length
            else:
                rate = epoch / length

            bar_length = self._bar_length
            marks = '#' * int(rate * bar_length)
            out.write('     total [{}{}] {:6.2%}\n'.format(
                marks, '.' * (bar_length - len(marks)), rate))

            epoch_rate = epoch - int(epoch)
            marks = '#' * int(epoch_rate * bar_length)
            out.write('this epoch [{}{}] {:6.2%}\n'.format(
                marks, '.' * (bar_length - len(marks)), epoch_rate))

            status = stat_template.format(trainer.updater)
            out.write(status)

            old_t, old_e, old_sec = recent_timing[0]
            span = now - old_sec
            if span != 0:
                speed_t = (iteration - old_t) / span
                speed_e = (epoch - old_e) / span
            else:
                speed_t = float('inf')
                speed_e = float('inf')

            if unit == 'iteration':
                estimated_time = (length - iteration) / speed_t
            else:
                estimated_time = (length - epoch) / speed_e
            out.write('{:10.5g} iters/sec. Estimated time to finish: {}.\n'
                      .format(speed_t,
                              datetime.timedelta(seconds=estimated_time)))

            # move the cursor to the head of the progress bar
            if os.name == 'nt':
                util.set_console_cursor_position(0, -4)
            else:
                out.write('\033[4A')
            out.flush()

            if len(recent_timing) > 100:
                del recent_timing[0]

    def finalize(self):
        # delete the progress bar
        out = self._out
        if os.name == 'nt':
            util.erase_console(0, 0)
        else:
            out.write('\033[J')
        out.flush()
