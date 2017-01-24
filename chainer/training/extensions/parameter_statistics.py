from chainer import reporter
import chainer.training as training
from chainer.training import extension
import statistics


def prefix_statistic_keys(key_prefix, stats):
    for key in list(stats.keys()):
        stats['{}/{}'.format(key_prefix, key)] = stats.pop(key)


class ParameterStatistics(extension.Extension):

    """Trainer extension to report parameter statistics.

    The statistics are collected for a given `~chainer.Link` or an iterable of
    `~chainer.Link`s. If a link contains child links, the statistics are
    aggregated over all its children.

    Statistics that can be collected and reporter using the current scope are
    as follows. However, the list may extend to other statistics depending on
    the type of parameter container.

    - Weight percentiles.
    - Bias percentiles.
    - Weight gradient percentiles.
    - Bias gradients percentiles.
    - Sparsity (counting number of zeros).

    Args:
        links (~chainer.Link or iterable of ~chainer.Link): Links containing
            the parameters to observe. The link is expected to have a ``name``
            attribute which is used as a part of a key in the report.
        trigger: Trigger that decides when to aggregate the results and report
            the values.
        sparsity (bool): If ``True``, include sparsity statistics.
        sparsity_include_bias (bool): If ``True``, take biases into account
            when computing the sparsity statistics. Otherwise, only consider
            weights. Does nothing if ``sparsity`` is ``False``.
        prefix (str): Prefix to prepend to the report keys.
    """

    default_name = 'parameter_statistics'
    priority = extension.PRIORITY_WRITER

    def __init__(self, links, trigger=(1, 'epoch'), sparsity=True,
                 sparsity_include_bias=True, prefix=None):

        if not isinstance(links, (tuple, list)):
            links = links,

        self._links = links
        self._trigger = training.trigger.get_trigger(trigger)
        self._prefix = prefix
        self._summary = reporter.DictSummary()
        self._targets = [('W', 'data'), ('b', 'data'),
                         ('W', 'grad'), ('b', 'grad')]
        self._sparsity_targets = []

        if sparsity:
            if sparsity_include_bias:
                self._sparsity_targets.append((('W', 'b'), 'data'))
            else:
                self._sparsity_targets.append((('W'), 'data'))

    def __call__(self, trainer):

        """Execute the extension and collect statistics for the current state
        of parameters.

        Note that this method will merely update its statistic summary, unless
        the internal trigger is fired. If the trigger is fired, the summary
        will also be reported and then reset for the next accumulation.

        Args:
            trainer (~chainer.training.Trainer): Associated trainer that
                invoked this extension.
        """

        for link in self._links:
            for target in self._targets:
                stats = self.get_statistics(link, *target)
                stats = self.post_process(stats)
                self._summary.add(stats)

            for target in self._sparsity_targets:
                stats = self.get_sparsity(link, *target)
                stats = self.post_process(stats)
                self._summary.add(stats)

        if self._trigger(trainer):
            reporter.report(self._summary.compute_mean())
            self._summary = reporter.DictSummary()  # Clear summary

    def post_process(self, stats):
        if self._prefix is not None:
            prefix_statistic_keys(self._prefix, stats)
        return stats

    def get_statistics(self, link, param_names, attr_names):
        return statistics.get_statistics(link, param_names, attr_names)

    def get_sparsity(self, link, param_names, attr_names):
        return statistics.get_sparsity(link, param_names, attr_names)
