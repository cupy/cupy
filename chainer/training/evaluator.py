import copy
import six

from chainer import link
from chainer import reporter as reporter_module
from chainer import variable
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module


class Evaluator(object):
    """Base class of all evaluators.
    """

    def run(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def get_iterator(self, name):
        raise NotImplementedError

    def get_all_iterators(self):
        raise NotImplementedError

    def get_target(self, name):
        raise NotImplementedError

    def get_all_targets(self):
        raise NotImplementedError


def eval_func_with_volatile(eval_func, in_arrays):
    if isinstance(in_arrays, tuple):
        in_vars = tuple(
                variable.Variable(x, volatile='on') for x in in_arrays
        )
        eval_func(*in_vars)
    elif isinstance(in_arrays, dict):
        in_vars = {
            key: variable.Variable(x, volatile='on')
            for key, x in six.iteritems(in_arrays)
        }
        eval_func(**in_vars)
    else:
        in_var = variable.Variable(in_arrays, volatile='on')
        eval_func(in_var)


class StandardEvaluator(Evaluator):
    """Standard implementation of Evaluator.

    This is the standard implementation of :class:`Evaluator`.
    It evaluate *target link* with given iterator and summarize the reported
    values in the link by calling :func:`~chainer.reporter.report`.
    It accepts one or more iterators and one or more links.
    The default evaluate routine assumes that there is only one training
    dataset and one model. Users can override this evaluate routine by
    inheriting this class and overriding the :meth:`evaluate` method. Each
    batch is converted to input arrays by
    :func:`~chainer.datasets.concat_examples` by default,
    which can also manually set by ``converter`` argument.

        >>> from chainer.training.evaluator import StandardEvaluator
        >>> import chainer
        >>> import numpy as np
        >>>
        >>> model = chainer.links.Classifier(
        >>>         chainer.links.Linear(
        >>>                 2, 2,
        >>>                 nobias=True,
        >>>                 initialW=np.identity(2, dtype=np.float32),
        >>>         )
        >>> )
        >>>
        >>> data = [
        >>>     (np.asarray([1, 0], dtype=np.float32), np.int32(0)),  # true
        >>>     (np.asarray([1, 0], dtype=np.float32), np.int32(0)),  # true
        >>>     (np.asarray([0, 1], dtype=np.float32), np.int32(1)),  # true
        >>>     (np.asarray([0, 1], dtype=np.float32), np.int32(1)),  # true
        >>>     (np.asarray([0, 1], dtype=np.float32), np.int32(0)),  # false
        >>> ]
        >>>
        >>> iterator = chainer.iterators.SerialIterator(data, 5, repeat=False)
        >>>
        >>> evaluator = StandardEvaluator(iterator, model)
        >>> result = evaluator.run()
        >>> print(result)
        {'main/accuracy': 0.8000000119209289, 'main/loss': 0.5132616162300109}

    Args:
        iterator: Dataset iterator. It can also be a dictionary of iterators.
            If this is just an iterator, then the iterator is registered by
            the name ``'main'``.
        target: Target link object. It can also be a dictionary of links. If
            this is just a link, then the link is registered by the name
            ``'main'``.
        converter: Converter function to build input arrays. Each batch
            extracted by the main iterator and the ``device`` option are passed
            to this function. :func:`~chainer.dataset.concat_examples` is used
            by default.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        loss_func: Loss function. The target link of the main optimizer is used
            by default.
        eval_hook_before: Function to prepare for each evaluation process.
            It is called at the beginning of the evaluation.
            (ex. clear state of LSTM, set testing flag of batch normalization).
        eval_hook_after: Function to finish for each evaluation process. It is
            called at the end of the evaluation.
        prefix: Prefix observer name for the reporter.
    """

    def __init__(self, iterator, target,
                 converter=convert.concat_examples,
                 device=None, eval_func=None,
                 eval_hook_before=None, eval_hook_after=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}

        if isinstance(target, link.Link):
            target = {'main': target}

        self._iterators = iterator
        self._targets = target

        self.converter = converter
        self.device = device
        self.eval_hook_before = eval_hook_before
        self.eval_hook_after = eval_hook_after
        self.eval_func = eval_func

    def run(self, prefix=''):
        reporter = reporter_module.Reporter()
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))
        with reporter:
            result = self.evaluate()

        return result

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the given dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Users can override this method to customize the evaluation routine.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
                :func:`~chainer.report` without specifying any observer.

        """
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook_before:
            self.eval_hook_before(self)

        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)

                eval_func_with_volatile(eval_func, in_arrays)

            summary.add(observation)

        if self.eval_hook_after:
            self.eval_hook_after(self)

        return summary.compute_mean()

    def get_iterator(self, name):
        """Returns the iterator of the given name."""
        return self._iterators[name]

    def get_all_iterators(self):
        """Returns a dictionary of all iterators."""
        return dict(self._iterators)

    def get_target(self, name):
        """Returns the target link of the given name."""
        return self._targets[name]

    def get_all_targets(self):
        """Returns a dictionary of all target links."""
        return dict(self._targets)
