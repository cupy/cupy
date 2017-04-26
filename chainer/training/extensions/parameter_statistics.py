import numpy
import six

from chainer import reporter
from chainer import training
from chainer.training import extension


class ParameterStatistics(extension.Extension):
    """Trainer extension to report parameter statistics.

    Statistics are collected and reported for a given :class:`~chainer.Link`
    or an iterable of :class:`~chainer.Link`s. If a link contains child links,
    the statistics are reported separately for each child.

    Any function that takes a one-dimensional :class:`numpy.ndarray` or a
    :class:`cupy.ndarray` and outputs a single or multiple real numbers can be
    registered to handle the collection of statistics, e.g.
    :meth:`numpy.ndarray.mean`.

    The keys of reported statistics follow the convention of link name
    followed by parameter name, attribute name and function name, e.g.
    ``VGG16Layers/conv1_1/W/data/mean``. They are prepended with an optional
    prefix and appended with integer indices if the statistics generating
    function return multiple values.

    Args:
        links (~chainer.Link or iterable of ~chainer.Link): Link(s) containing
            the parameters to observe. The link is expected to have a ``name``
            attribute which is used as a part of the report key.
        statistics (dict): Dictionary with function name to function mappings.
            The name is a string and is used as a part of the report key. The
            function is responsible for generating the statistics.
        report_params (bool): If ``True``, report statistics for parameter
            values such as weights and biases.
        report_grads (bool): If ``True``, report statistics for parameter
            gradients.
        prefix (str): Optional prefix to prepend to the report keys.
        trigger: Trigger that decides when to aggregate the results and report
            the values.
    """
    default_name = 'parameter_statistics'
    priority = extension.PRIORITY_WRITER

    # prefix ends with a '/' and param_name is preceded by a '/'
    report_key_template = ('{prefix}{link_name}{param_name}/{attr_name}/'
                           '{function_name}')

    default_statistics = {
        'mean': numpy.mean,
        'std': numpy.std,
        'min': numpy.min,
        'max': numpy.max,
        'zeros': lambda x: numpy.count_nonzero(x == 0),
        'percentile': lambda x: numpy.percentile(x, (0.13, 2.28, 15.87,
                                                     50, 84.13, 97.72,
                                                     99.87))
    }

    def __init__(self, links, statistics=default_statistics,
                 report_params=True, report_grads=True, prefix=None,
                 trigger=(1, 'epoch')):

        if not isinstance(links, (list, tuple)):
            links = links,
        self._links = links

        self._statistics = statistics

        attrs = []
        if report_params:
            attrs.append('data')
        if report_grads:
            attrs.append('grad')
        self._attrs = attrs

        self._prefix = prefix
        self._trigger = training.trigger.get_trigger(trigger)
        self._summary = reporter.DictSummary()

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
        statistics = {}

        for link in self._links:
            link_name = getattr(link, 'name', 'None')
            for param_name, param in link.namedparams():
                for attr_name in self._attrs:
                    for function_name, function in \
                            six.iteritems(self._statistics):
                        # Get parameters as a flattend one-dimensional array
                        # since the statistics function should make no
                        # assumption about the axes
                        params = getattr(param, attr_name).ravel()
                        value = function(params)
                        key = self.report_key_template.format(
                            prefix=self._prefix + '/' if self._prefix else '',
                            link_name=link_name,
                            param_name=param_name,
                            attr_name=attr_name,
                            function_name=function_name
                        )
                        if hasattr(value, '__iter__'):
                            # Append integer indices to the keys if the
                            # statistic function return multiple values
                            statistics.update({'{}/{}'.format(key, i): v for
                                               i, v in enumerate(value)})
                        else:
                            statistics[key] = value

        self._summary.add(statistics)

        if self._trigger(trainer):
            reporter.report(self._summary.compute_mean())
            self._summary = reporter.DictSummary()  # Clear summary

    def register_statistics(self, name, function):
        """Register a function to compute a certain statistic.

        The registered function will be called each time the extension runs and
        the results will be included in the report.

        Args:
            name (str): Name of the statistic.
            function: Function to generate the statistic. Any function that
                takes a one-dimensional :class:`numpy.ndarray` or a
                :class:`cupy.ndarray` and outputs a single or multiple real
                numbers is allowed.
        """
        self._statistics[name] = function
