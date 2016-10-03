from chainer.dataset import convert
from chainer import reporter as reporter_module
from chainer.training import extension, StandardEvaluator


class Evaluator(extension.Extension):
    """Trainer extension to evaluate models on a validation set.

    This extension evaluates the current models by a given evaluation function.
    It creates a :class:`~chainer.Reporter` object to store values observed in
    the evaluation function on each iteration. The report for all iterations
    are aggregated to :class:`~chainer.DictSummary`. The collected mean values
    are further reported to the reporter object of the trainer, where the name
    of each observation is prefixed by the evaluator name. See
    :class:`~chainer.Reporter` for details in naming rules of the reports.

    Evaluator has a structure to customize similar to that of
    :class:`~chainer.training.StandardUpdater`. The main differences are:

    - There are no optimizers in an evaluator. Instead, it holds links
      to evaluate.
    - An evaluation loop function is used instead of an update function.
    - Preparation routine can be customized, which is called before each
      evaluation. It can be used, e.g., to initialize the state of stateful
      recurrent networks.

    There are two ways to modify the evaluation behavior besides setting a
    custom evaluation function. One is by setting a custom evaluation loop via
    the ``eval_func`` argument. The other is by inheriting this class and
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
        converter: Converter function to build input arrays.
            :func:`~chainer.dataset.concat_examples` is used by default.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        evaluator: Customizable Evaluator instance. If it is applied, all
            initial arguments are ignored and given evaluator is used.

    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.

    """
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, evaluator=None):
        if evaluator is None:
            evaluator = StandardEvaluator(
                    iterator=iterator, target=target,
                    converter=converter, device=device,
                    eval_hook_before=eval_hook, eval_func=eval_func,
            )
        self.evaluator = evaluator

    def get_iterator(self, name):
        """Returns the iterator of the given name."""
        return self.evaluator.get_iterator(name)

    def get_all_iterators(self):
        """Returns a dictionary of all iterators."""
        return self.evaluator.get_all_iterators()

    def get_target(self, name):
        """Returns the target link of the given name."""
        return self.evaluator.get_target(name)

    def get_all_targets(self):
        """Returns a dictionary of all target links."""
        return self.evaluator.get_all_targets()

    def __call__(self, trainer=None):
        """Executes the evaluator extension.

        This extension reports the performance on validation dataset using
        the :func:`~chainer.report` function.

        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.

        Returns:
            dict: Result dictionary that contains mean statistics of values
                reported by the evaluation function.

        """
        if hasattr(self, 'name'):
            prefix = self.name + '/'
        else:
            prefix = ''

        result = self.evaluator.run(prefix)

        reporter_module.report(result)
        return result
