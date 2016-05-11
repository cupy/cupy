import contextlib


class Report(object):

    """Object to which observed values are reported.

    Report is used to collect values that users want to watch. The report
    object holds a mapping from value names to the actually observed values.

    When a value is passed to the report, an observer object can be optionally
    attached. In this case, the name of the observer is added as the prefix of
    the value name, separated with a single colon ``:``. The observer name
    should be registered beforehand.

    See the following example::

       >>> from chainer.trainer import Report, report
       >>>
       >>> r = Report()
       >>> observer = object()  # it can be an arbitrary (reference) object
       >>> r.add_observer('my_observer', observer)
       >>> observation = {}
       >>> with r.scope(observation):
       ...     r.report('x', 1, observer)
       ...
       >>> observation
       {'my_observer:x': 1}

    There are also a global API to add values::

       >>> observation = {}
       >>> with r.scope(observation):
       ...     report('x', 1, observer)

    Attributes:
        observation: Dictionary of observed values.

    """
    def __init__(self):
        self._observer_names = {}
        self.observation = {}

    def __enter__(self):
        """Makes this report object current."""
        _reports.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        """Recovers the previous report object to the current."""
        _reports.pop()

    @contextlib.contextmanager
    def scope(self, observation):
        """Creates a scope to report observed values to the given dictionary.

        This is a context manager to be passed to ``with`` statements. In this
        scope, the observation dictionary is changed to the given one.

        It also makes this report object current.

        Args:
            observation (dict): Observation dictionary. All observations
                registered to this observer inside of the ``with`` statement
                are written to this dictionary

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
            observer: The observer object. Note that the observer distinguishes
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

    def write(self, name, value, observer=None):
        """Writes an observed value to this report.

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


_reports = [Report()]


def get_current_report():
    """Returns the current report object."""
    return _reports[-1]


def report(name, value, observer=None):
    """Reports an observed value to the current report object.

    Any report object can be set current by the ``with`` statement. This
    function calls the :meth:`Report.write` method of the current report.

    Args:
        name (str): Name of the value.
        value: Observed value.
        observer: Observer object. Its object ID is used to retrieve the
            observer name, which is used as the prefix of the registration name
            of the observed value.

    """
    current = _reports[-1]
    current.write(name, value, observer)
