import numpy
import six

from chainer import cuda
from chainer import reporter
from chainer import training
from chainer.training import extension


def _iterable(x):
    if isinstance(x, (list, tuple)):
        return x
    return x,


def _prefix_dict_keys(prefix, x):
    return {'{}/{}'.format(prefix, k): v for k, v in six.iteritems(x)}


def _target_name(link, param_names, attr_names):
    """Generate a dictionary key based on context."""
    link_name = getattr(link, 'name', 'None')
    param_name = '-'.join(_iterable(param_names))
    attr_name = '-'.join(_iterable(attr_names))

    return '{}/{}/{}'.format(link_name, param_name, attr_name)


def _get_link_params(link, param_names, attr_names):
    """Flatten link parameters into a single array and return a copy."""
    params = []
    for param in link.params():
        if param.name in _iterable(param_names):
            for attr_name in _iterable(attr_names):
                params.append(getattr(param, attr_name).ravel())

    return link.xp.concatenate(params) if params else link.xp.array([])


def _statistics(x, functions):
    """Compute statisticts for the given array.

    Args:
        x (array): Target array for which statistics are computed.
        functions (iterable): Statistics to collect, mapping directly to NumPy
            or CuPy functions.

    Returns:
        dict: Mappings from functions keys to statistic values.
    """
    stats = {}
    for f in functions:
        try:
            # nan if x.size == 0 and f in ('mean', 'std')
            stats[f] = getattr(x, f, lambda: float('NaN'))()
        except ValueError:  # x.size == 0 and f in ('min, 'max')
            stats[f] = float('NaN')

    return stats


def _percentiles(x, sigmas):
    """Compute percentiles for the given array.

    Args:
        x (array): Target array for which percentiles are computed.
        sigmas (iterable): Percentile sigma values.

    Returns:
        array: List of percentiles. The list has the same length as the given
            ``sigma``.
    """
    def _percentiles_cpu(_x):
        try:
            return numpy.percentile(_x, sigmas).astype(_x.dtype)
        except IndexError:  # _x.size == 0
            return numpy.array((float('NaN'),) * len(sigmas))

    if cuda.available and isinstance(x, cuda.ndarray):
        # cuda.cupy.percentile() is not implemented
        return cuda.to_gpu(_percentiles_cpu(cuda.to_cpu(x)))

    return _percentiles_cpu(x)


def _zeros(x):
    """Count the number of zeros in the given array.

    Args:
        x (array): Target array for which sparsity is computed.

    Returns:
        int: Number of zeros.
    """
    if x.ndim == 0:
        raise ValueError('Cannot count zeros for shape {}'.format(x.shape))

    return x.size - cuda.get_array_module(x).count_nonzero(x)


class ParameterStatistics(extension.Extension):
    """Trainer extension to report parameter statistics.

    The statistics are collected for a given `~chainer.Link` or an iterable of
    `~chainer.Link`s. If a link contains child links, the statistics are
    aggregated over all its children.

    Statistics that can be collected and reporter using the current scope are
    minimum and maximum values, means, standard deviations, percentiles and
    zero counts.

    Args:
        links (~chainer.Link or iterable of ~chainer.Link): Links containing
            the parameters to observe. The link is expected to have a ``name``
            attribute which is used as a part of a key in the report.
        trigger: Trigger that decides when to aggregate the results and report
            the values.
        count_zeros (bool): If ``True``, count the number of zero elements and
            include those statistics in the report. Else, do not compute the
            number of zero elements.
        prefix (str): Prefix to prepend to the report keys.
        targets (iterable of tuples): Target parameters and their attributes
            to consider in this extension.
    """
    default_name = 'parameter_statistics'
    priority = extension.PRIORITY_WRITER

    def __init__(self, links, trigger=(1, 'epoch'), count_zeros=False,
                 prefix=None, targets=(('W', 'data'),
                                       ('b', 'data'),
                                       ('W', 'grad'),
                                       ('b', 'grad'))):

        self._links = _iterable(links)
        self._trigger = training.trigger.get_trigger(trigger)
        self._count_zeros = count_zeros
        self._prefix = prefix
        self._targets = targets

        self._summary = reporter.DictSummary()

        self._statistic_functions = ('min', 'max', 'mean', 'std')
        self._percentile_sigmas = (0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87)

    def __call__(self, trainer):
        """Execute the statistics extension.

        Collect statistics for the current state of parameters.

        Note that this method will merely update its statistic summary, unless
        the internal trigger is fired. If the trigger is fired, the summary
        will also be reported and then reset for the next accumulation.

        Args:
            trainer (~chainer.training.Trainer): Associated trainer that
                invoked this extension.
        """
        for link in self._links:
            for target in self._targets:
                params = _get_link_params(link, *target)
                if params.size > 0:
                    prefix = _target_name(link, *target) + '/'
                    stats = self.statistics_report(params, prefix)
                    stats.update(self.percentiles_report(params, prefix))
                    if self._count_zeros:
                        stats.update(self.zeros_report(params, prefix))
                    self._summary.add(self.post_process(stats))

        if self._trigger(trainer):
            reporter.report(self._summary.compute_mean())
            self._summary = reporter.DictSummary()  # Clear summary

    def statistics_report(self, params, prefix=''):
        return {'{}{}'.format(prefix, f): s for f, s in
                six.iteritems(_statistics(params, self._statistic_functions))}

    def percentiles_report(self, params, prefix=''):
        return {'{}percentile/{}'.format(prefix, i): p for i, p
                in enumerate(_percentiles(params, self._percentile_sigmas))}

    def zeros_report(self, params, prefix=''):
        return {'{}zeros'.format(prefix): _zeros(params)}

    def post_process(self, stats):
        if self._prefix is not None:
            stats = _prefix_dict_keys(self._prefix, stats)
        return stats
