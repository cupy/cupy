import six

from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module
from chainer import variable


class Updater(object):

    """Interface of updater objects for trainers.

    TODO(beam2d): document it.

    """
    def finalize(self):
        """Finalizes the updater object.

        This method is called at the end of training loops. It should finalize
        each dataset iterator used in this updater.

        """
        raise NotImplementedError

    def get_optimizer(self, name):
        """Gets the optimizer of given name.

        Updater holds one or more optimizers with names. They can be retrieved
        by this method.

        Args:
            name (str): Name of the optimizer.

        Returns:
            ~chainer.Optimizer: Optimizer of the name.

        """
        raise NotImplementedError

    def get_all_optimizers(self):
        """Gets a dictionary of all optimizers for this updater.

        Returns:
            dict: Dictionary that maps names to optimizers.

        """
        raise NotImplementedError

    def update(self):
        """Updates the parameters of the target model.

        This method implements an update formula for the training task,
        including data loading, forward/backward computations, and actual
        updates of parameters.

        This method is called once at each iteration of the training loop.

        """
        raise NotImplementedError

    def serialize(self, serializer):
        """Serializes the current state of the updater object."""
        raise NotImplementedError


class StandardUpdater(Updater):

    """Standard implementation of Updater.

    This is the standard implementation of :class:`Updater`. It accepts one or
    more training datasets and one or more optimizers. The default update
    routine assumes that there is only one training dataset and one optimizer,
    while users can specify their own update routines. Each batch is converted
    to input arrays by :func:`~chainer.datasets.concat_examples` by default,
    which can also be manually set.

    There are two ways to modify the update behavior besides setting a custom
    loss function. One is by setting a custom update function via the
    ``update_func`` argument. The other one is by inheriting this class and
    overriding the :meth:`update` method. In latter case, do not forget to
    update the iteration counter at each call of this method, because this
    value is watched by the trainer for deciding when to invoke extensions and
    when to exit the training loop.

    Args:
        iterator: Dataset iterator for the training dataset. It can also be a
            dictionary of iterators. If this is just an iterator, then the
            iterator is registered by the name ``'main'``.
        optimizer: Optimizer to update parameters. It can also be a dictionary
            of optimizers. If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        update_func: Update routine. This is a function that takes the updater
            object as the argument. The default routine uses ``converter`` and
            ``loss_func`` if specified.
        converter: Converter function to build input arrays. Each batch
            extracted by the main iterator and the ``device`` option are passed
            to this function. If it is omitted,
            :func:`~chainer.dataset.concat_examples` is used. If
            ``update_func`` is specified, this argument is ignored and not
            used.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU). If ``update_func`` is specified,
            this argument is ignored and not used.
        loss_func: Loss function. The target link of the main optimizer is used
            by default. If ``update_func`` is specified, this argument is
            ignored and not used.

    Attributes:
        iteration: Current number of completed updates.

    """
    def __init__(self, iterator, optimizer, update_func=None, converter=None,
                 device=None, loss_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(optimizer, optimizer_module.Optimizer):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        self._update_func = update_func or _default_update(
            self, converter, device, loss_func)

        self.iteration = 0

    @property
    def epoch(self):
        return self._iterators['main'].epoch

    @property
    def epoch_detail(self):
        return self._iterators['main'].epoch_detail

    @property
    def is_new_epoch(self):
        return self._iterators['main'].is_new_epoch

    def finalize(self):
        for iterator in six.itervalues(self._iterators):
            iterator.finalize()

    def get_optimizer(self, name):
        return self._optimizers[name]

    def get_all_optimizers(self):
        return dict(self._optimizers)

    def get_iterator(self, name):
        """Gets the dataset iterator of given name.

        Args:
            name (str): Name of the dataset iterator.

        Returns:
            ~chainer.dataset.Iterator: Corresponding dataset iterator.

        """
        return self._iterators[name]

    def update(self):
        self._update_func(self)
        self.iteration += 1

    def serialize(self, serializer):
        for name, iterator in six.iteritems(self._iterators):
            iterator.serialize(serializer['iterator:' + name])

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.serialize(serializer['optimizer:' + name])
            optimizer.target.serialize(serializer['model:' + name])

        self.iteration = serializer('iteration', self.iteration)


def _default_update(updater, converter, device, loss_func):
    if not converter:
        converter = convert.concat_examples

    iterator = updater.get_iterator('main')
    optimizer = updater.get_optimizer('main')
    loss_func = loss_func or optimizer.target

    def update(_):
        batch = iterator.next()
        in_arrays = converter(batch, device)

        if isinstance(in_arrays, tuple):
            in_vars = tuple(variable.Variable(x) for x in in_arrays)
            optimizer.update(loss_func, *in_vars)
        elif isinstance(in_arrays, dict):
            in_vars = {key: variable.Variable(x)
                       for key, x in six.iteritems(in_arrays)}
            optimizer.update(loss_func, **in_vars)
        else:
            in_var = variable.Variable(in_arrays)
            optimizer.update(loss_func, in_var)

    return update
