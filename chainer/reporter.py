import collections
import contextlib

import numpy
import six

from chainer import cuda
from chainer import variable


class Reporter(object):

    """Object to which observed values are reported.

    Reporter is used to collect values that users want to watch. The reporter
    object holds a mapping from value names to the actually observed values.

    When a value is passed to the reporter, an observer object can be
    optionally attached. In this case, the name of the observer is added as the
    prefix of the value name, separated with a single colon ``:``. The observer
    name should be registered beforehand.

    See the following example::

       >>> from chainer import Reporter, report, report_scope
       >>>
       >>> reporter = Reporter()
       >>> observer = object()  # it can be an arbitrary (reference) object
       >>> reporter.add_observer('my_observer', observer)
       >>> observation = {}
       >>> with reporter.scope(observation):
       ...     reporter.report('x', 1, observer)
       ...
       >>> observation
       {'my_observer:x': 1}

    There are also a global API to add values::

       >>> observation = {}
       >>> with report_scope(observation):
       ...     report('x', 1, observer)

    Attributes:
        observation: Dictionary of observed values.

    """
    def __init__(self):
        self._observer_names = {}
        self.observation = {}

    def __enter__(self):
        """Makes this reporter object current."""
        _reporters.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        """Recovers the previous reporter object to the current."""
        _reporters.pop()

    @contextlib.contextmanager
    def scope(self, observation):
        """Creates a scope to report observed values to the given dictionary.

        This is a context manager to be passed to ``with`` statements. In this
        scope, the observation dictionary is changed to the given one.

        It also makes this reporter object current.

        Args:
            observation (dict): Observation dictionary. All observations
                reported inside of the ``with`` statement are written to this
                dictionary.

        """
        old = self.observation
        self.observation = observation
        self.__enter__()
        yield
        self.__exit__(None, None, None)
        self.observation = old

    def add_observer(self, name, observer):
        """Registers an observer of values.

        Observer defines a scope of names for observed values. Values observed
        with the observer are registered with names prefixed by the observer
        name.

        Args:
            name (str): Name of the observer.
            observer: The observer object. Note that the reporter distinguishes
                the observers by their object ids (i.e., ``id(owner)``), rather
                than the object equality.

        """
        self._observer_names[id(observer)] = name

    def add_observers(self, prefix, observers):
        """Registers multiple observers at once.

        This is a convenient method to register multiple objects at once.

        Args:
            prefix (str): Prefix of each name of observers.
            observers: Iterator of name and observer pairs.

        """
        for name, observer in observers:
            self._observer_names[id(observer)] = prefix + name

    def report(self, name, value, observer=None):
        """Reports an observed value.

        The value is written with the given name, prefixed by the name of the
        observer object if given.

        Args:
            name (str): Name of the value.
            value: Observed value.
            observer: Observer object. Its object ID is used to retrieve the
                observer name, which is used as the prefix of the registration
                name of the observed value.

        """
        if observer is not None:
            observer_name = self._observer_names[id(observer)]
            name = '%s:%s' % (name, observer_name)

        self.observation[name] = value


_reporters = [Report()]


def get_current_reporter():
    """Returns the current reporter object."""
    return _reporters[-1]


def report(name, value, observer=None):
    """Reports an observed value with the current reporter object.

    Any reporter object can be set current by the ``with`` statement. This
    function calls the :meth:`Report.write` method of the current report.

    Args:
        name (str): Name of the value.
        value: Observed value.
        observer: Observer object. Its object ID is used to retrieve the
            observer name, which is used as the prefix of the registration name
            of the observed value.

    """
    current = _reporters[-1]
    current.report(name, value, observer)


def report_scope(observation):
    """Returns a report scope with the current reporter.

    This is equivalant to ``get_current_reporter().report_scope(observation)``.

    """
    current = _reporters[-1]
    return current.report_scope(observation)


class Summary(object):

    """Online summarization of a sequence of scalars.

    Summary computes the statistics of given scalars online.

    """
    def __init__(self):
        self._x = 0
        self._x2 = 0
        self._n = 0

    def add(self, value):
        """Adds a scalar value.

        Args:
            value: Scalar value to accumulate. It is either a NumPy scalar or
                a zero-dimensional array (on CPU or GPU).

        """
        with cuda.get_device(value):
            self._x += value
            self._x2 += value * value
            self._n += 1

    @property
    def mean(self):
        """Mean of given scalars."""
        x = self._x
        with cuda.get_device(x):
            return x / self._n

    @property
    def variance(self):
        """Variance of given scalars."""
        x, n = self._x, self._n
        with cuda.get_device(x):
            mean = x / n
            return self._x2 / n - mean * mean

    @property
    def std(self):
        """Standard deviation of given scalars."""
        x, n = self._x, self._n
        xp = cuda.get_array_module(x)
        with cuda.get_device(x):
            mean = x / n
            var = self._x2 / n - mean * mean
            return xp.sqrt(var, out=var)


class DictSummary(object):

    """Online summarization of a sequence of dictionaries.

    DictSummary computes the statistics of a given set of scalars online. It
    only computes the statistics for scalar values and variables of scalar
    values in the dictionaries.

    """
    def __init__(self):
        self._summaries = collections.defaultdict(Summary)

    def add(self, d):
        """Adds a dictionary of scalars.

        Args:
            d (dict): Dictionary of scalars to accumulate. Only elements of
               scalars, zero-dimensional arrays, and variables of
               zero-dimensional arrays are accumulated.

        """
        summaries = self._summaries
        for k, v in six.iteritems(d):
            if isinstance(v, variable.Variable):
                v = v.data
            elif not isinstance(v, (numpy.ndarray, cuda.ndarray)):
                if numpy.isscalar(v):
                    summaries[k].add(v)
                else:
                    continue

            if v.ndim == 0:
                summaries[k].add(v)

    @property
    def mean(self):
        """A dictionary of the means of accumulated values.

        This property always returns a new dictionary.

        """
        return {k: s.mean for k, s in six.iteritems(self._summaries)}

    @property
    def variance(self):
        """A dictionary of the variances of accumulated values.

        This property always returns a new dictionary.

        """
        return {k: s.variance for k, s in six.iteritems(self._summaries)}

    @property
    def std(self):
        """A dictionary of the standard deviations of accumulated values.

        This property always returns a new dictionary.

        """
        return {k: s.std for k, s in six.iteritems(self._summaries)}
