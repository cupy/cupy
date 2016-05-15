import sys

from chainer.training import extension
from chainer.training.extensions import log_report as log_report_module


class PrintReport(extension.Extension):

    """Trainer extension to print the accumulated results.

    This extension uses the log accumulated by a :class:`LogReport` extension
    to print specified entries of the log in a human-readable format.

    Args:
        entries (list of str): List of keys of observations to print.
        log_report (str or LogReport): Log report to accumulate the
            observations. This is either the name of a LogReport extensions
            registered to the trainer, or a LogReport instance to use
            interanlly.
        out: Stream to print the bar. Standard output is used by default.

    """
    def __init__(self, entries, log_report='LogReport', out=sys.stdout):
        self._entries = entries
        self._log_report = log_report
        self._out = out

        self._log_len = 0  # number of observations already printed

        # format information
        entry_widths = [max(10, len(s)) for s in entries]

        header = '  '.join(('{:%d}' % w for w in entry_widths)).format(
            *entries) + '\n'
        self._header = header  # printed at the first call

        template = []
        for entry, w in zip(entries, entry_widths):
            template.append('{%s:<%dg}' % (entry, w))
        self._template = '  '.join(template) + '\n'

    def __call__(self, trainer):
        out = self._out

        # delete the printed contents from the current cursor
        out.write(u'\x1b\x9bJ')

        if self._header:
            out.write(self._header)
            self._header = None

        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)  # update the log report
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

        log = log_report.log
        log_len = self._log_len
        template = self._template
        while len(log) > log_len:
            out.write(template.format(**log[log_len]))
            log_len += 1
        self._log_len = log_len

    def serialize(self, serializer):
        log_report = self._log_report
        if isinstance(log_report, log_report_module.LogReport):
            log_report.serialize(serializer['_log_report'])
