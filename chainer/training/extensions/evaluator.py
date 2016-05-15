import copy

import six

from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extension
from chainer import variable


class Evaluator(extension.Extension):

    """Trainer extension to evaluate models on a validation set.

    This extension evaluates the current models by a given evaluation function.
    It creates a :class:`~chainer.Reporter` object to store values observed in
    the evaluation function on each iteration. The report for all iterations
    are aggregated to :class:`~chainer.DictSummary`. The collected mean values
    are further reported to the reporter object of the trainer, where the name
    of each observation is prefixed by the evaluator name. It also report their
    variances and standard deviations, where the names are also suffixed by
    ``.variance`` and ``.std``, respectively.

    Evaluator has a structure to customize similar to that of
    :class:`StandardUpdater`. The main differences are:

    - There are no optimizers in an evaluator. Instead, it holds link
      to evaluate.
    - An evaluation loop function is used instead of an update function.
    - Preparation routine can be customized, which is called before each
      evaluation. It can be used, e.g., to initialize the state of stateful
      networks.

    There are two ways to modify the evaluation behavior besides setting a
    custom evaluation function. One is by setting a custom evaluation loop via
    the ``eval_loop`` argument. The other is by inheriting this class and
    overriding the :meth:`evaluate` method. In latter case, users have to
    create and handle a reporter object manually. Users also have to copy the
    iterators before using them, in order to reuse them at the next time of
    evaluation.

    This extension is called at the end of each epoch by default.

    Args:
        iterator: Dataset iterator for the validation dataset. It can also be
            a dictionary of iterators. If this is just an iterator, the
            iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        eval_loop: Evaluation loop. This is a function that takes the evaluator
            object as the argument, and returns a dictionary of statistics for
            observed values. The default routine uses ``converter`` and
            ``eval_func`` if specified.
        converter: Converter function to build input arrays. If it is omitted,
            :func:`~chainer.dataset.concat_examples` is used. If
            ``eval_loop`` is specified, this argument is ignored and not used.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU). If ``eval_loop`` or ``converter``
            is specified, this argument is ignored and not used.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default. If ``eval_loop``
            is specified, this argument is ignored and not used.

    """
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, eval_loop=None, converter=None,
                 device=None, eval_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, link.Link):
            target = {'main': target}
        self._targets = target

        self._eval_loop = eval_loop or _default_eval_loop(
            self, converter, device, eval_func)

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

    def __call__(self, trainer):
        result = self.evaluate()
        trainer.observation.update(result)

    def evaluate(self):
        """Executes an evaluation.

        This method runs an evaluation and returns the statistics of the
        observation. It calls the evaluation loop and returns the collected
        statistics of observed values. During the evaluation loop, a reporter
        to which the registered links are set as observers is used.

        Returns:
            dict: Dictionary of statistics of observations made by the
                evaluation routines.

        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        prefix = self.name + '/'
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        with reporter:
            return self._eval_loop(self)


def _default_eval_loop(evaluator, converter, device, eval_func):
    if not converter:
        def _convert(batch):
            return convert.concat_examples(batch, device=device)
        converter = _convert

    iterator = evaluator.get_iterator('main')
    target = evaluator.get_target('main')
    eval_func = eval_func or target

    def eval_loop(trainer):
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = converter(batch)
                if isinstance(in_arrays, tuple):
                    in_vars = tuple(variable.Variable(x) for x in in_arrays)
                    eval_func(*in_vars)
                elif isinstance(in_arrays, dict):
                    in_vars = {key: variable.Variable(x)
                               for key, x in six.iteritems(in_arrays)}
                    eval_func(**in_vars)
                else:
                    in_var = variable.Variable(in_arrays)
                    eval_func(in_var)

            summary.add(observation)

        return summary.compute_mean()

    return eval_loop
