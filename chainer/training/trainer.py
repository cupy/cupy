import collections
import os
import time

import six

from chainer import reporter as reporter_module
from chainer import serializer as serializer_module
from chainer.training import extension as extension_module
from chainer.training import trigger as trigger_module


class _ExtensionEntry(object):

    def __init__(self, extension, priority, trigger, invoke_before_training):
        self.extension = extension
        self.trigger = trigger
        self.invoke_before_training = invoke_before_training
        self.priority = priority


class Trainer(object):

    """The standard training loop in Chainer.

    Trainer is an implementation of a training loop. Users can invoke the
    training by calling the :meth:`run` method.

    Each iteration of the training loop proceeds as follows.

    - Update of the parameters. It includes the mini-batch loading, forward
      and backward computations, and an execution of the update formula.
      These are all done by the update object held by the trainer.
    - Invocation of trainer extensions in the descending order of their
      priorities. A trigger object is attached to each extension, and it
      decides at each iteration whether the extension should be executed.
      Trigger objects are callable objects that take the trainer object as the
      argument and return a boolean value indicating whether the extension
      should be called or not.

    Extensions are callable objects that take the trainer object as the
    argument. There are three ways to define custom extensions: inheriting the
    :class:`Extension` class, decorating functions by :func:`make_extension`,
    and defining any callable including lambda functions. See
    :class:`Extension` for more details on custom extensions and how to
    configure them.

    Users can register extensions to the trainer by calling the :meth:`extend`
    method, where some configurations can be added.

    - Trigger object, which is also explained above. In most cases,
      :class:`IntervalTrigger` is used, in which case users can simply specify
      a tuple of the interval length and its unit, like
      ``(1000, 'iteration')`` or ``(1, 'epoch')``.
    - The order of execution of extensions is determined by their priorities.
      Extensions of higher priorities are invoked earlier. There are three
      standard values for the priorities:

      - ``PRIORITY_WRITER``. This is the priority for extensions that write
        some records to the :attr:`observation` dictionary. It includes cases
        that the extension directly adds values to the observation dictionary,
        or the extension uses the :func:`chainer.report` function to report
        values to the observation dictionary.
      - ``PRIORITY_EDITOR``. This is the priority for extensions that edit the
        :attr:`observation` dictionary based on already reported values.
      - ``PRIORITY_READER``. This is the priority for extensions that only read
        records from the :attr:`observation` dictionary. This is also suitable
        for extensions that do not use the :attr:`observation` dictionary at
        all.

    - Extensions with ``invoke_before_training`` flag on are also invoked at
      the beginning of the training loop. Extensions that update the training
      status (e.g., changing learning rates) should have this flag to be
      ``True`` to ensure that resume of the training loop correctly recovers
      the training status.

    The current state of the trainer object and objects handled by the trainer
    can be serialized through the standard serialization protocol of Chainer.
    It enables us to easily suspend and resume the training loop.

    .. note::
       The serialization does not recover everything of the training loop. It
       only recovers the states which change over the training (e.g.
       parameters, optimizer states, the batch iterator state, extension
       states, etc.). You must initialize the objects correctly before
       deserializing the states.

       On the other hand, it means that users can change the settings on
       deserialization. For example, the exit condition can be changed on the
       deserialization, so users can train the model for some iterations,
       suspend it, and then resume it with larger number of total iterations.

    During the training, it also creates a :class:`~chainer.Reporter` object to
    store observed values on each update. For each iteration, it creates a
    fresh observation dictionary and stores it in the :attr:`observation`
    attribute.

    Links of the target model of each optimizer are registered to the reporter
    object as observers, where the name of each observer is constructed as the
    format ``<optimizer name><link name>``. The link name is given by the
    :meth:`chainer.Link.namedlink` method, which represents the path to each
    link in the hierarchy. Other observers can be registered by accessing the
    reporter object via the :attr:`reporter` attribute.

    The default trainer is `plain`, i.e., it does not contain any extensions.

    Args:
        updater (~chainer.training.Updater): Updater object. It defines how to
            update the models.
        stop_trigger: Trigger that determines when to stop the training loop.
            If it is not callable, it is passed to :class:`IntervalTrigger`.

    Attributes:
        updater: The updater object for this trainer.
        stop_trigger: Trigger that determines when to stop the training loop.
            The training loop stops at the iteration on which this trigger
            returns ``True``.
        observation: Observation of values made at the last update. See the
            :class:`Reporter` class for details.
        out: Output directory.
        reporter: Reporter object to report observed values.

    """

    def __init__(self, updater, stop_trigger=None, out='result'):
        self.updater = updater
        self.stop_trigger = trigger_module.get_trigger(stop_trigger)
        self.observation = {}
        self.out = out

        reporter = reporter_module.Reporter()
        for name, optimizer in six.iteritems(updater.get_all_optimizers()):
            reporter.add_observer(name, optimizer.target)
            reporter.add_observers(
                name, optimizer.target.namedlinks(skipself=True))
        self.reporter = reporter

        self._done = False
        self._extensions = collections.OrderedDict()

        self._start_at = None
        self._snapshot_elapsed_time = 0.0
        self._final_elapsed_time = None

        updater.connect_trainer(self)

    @property
    def elapsed_time(self):
        """Total time used for the training.

        The time is in seconds. If the training is resumed from snapshot, it
        includes the time of all the previous training to get the current
        state of the trainer.

        """
        if self._done:
            return self._final_elapsed_time
        if self._start_at is None:
            raise RuntimeError('training has not been started yet')
        return time.time() - self._start_at + self._snapshot_elapsed_time

    def extend(self, extension, name=None, trigger=None, priority=None,
               invoke_before_training=None):
        """Registers an extension to the trainer.

        :class:`Extension` is a callable object which is called after each
        update unless the corresponding trigger object decides to skip the
        iteration. The order of execution is determined by priorities:
        extensions with higher priorities are called earlier in each iteration.
        Extensions with the same priority are invoked in the order of
        registrations.

        If two or more extensions with the same name are registered, suffixes
        are added to the names of the second to last extensions. The suffix is
        ``_N`` where N is the ordinal of the extensions.

        See :class:`Extension` for the interface of extensions.

        Args:
            extension: Extension to register.
            name (str): Name of the extension. If it is omitted, the
                ``default_name`` attribute of the extension is used instead.
                Note that the name would be suffixed by an ordinal in case of
                duplicated names as explained above.
            trigger (tuple or Trigger): Trigger object that determines when to
                invoke the extension. If it is ``None``, ``extension.trigger``
                is used instead. If it is ``None`` and the extension does not
                have the trigger attribute, the extension is triggered at every
                iteration by default. If the trigger is not callable, it is
                passed to :class:`IntervalTrigger` to build an interval
                trigger.
            priority (int): Invocation priority of the extension. Extensions
                are invoked in the descending order of priorities in each
                iteration. If this is ``None``, ``extension.priority`` is used
                instead.
            invoke_before_training (bool or None): If ``True``, the extension
                is also invoked just before entering the training loop. If this
                is ``None``, ``extension.invoke_before_training`` is used
                instead. This option is mainly used for extensions that alter
                the training configuration (e.g., learning rates); in such a
                case, resuming from snapshots require the call of extension to
                recover the configuration before any updates.

        """
        if name is None:
            name = getattr(extension, 'name', None)
            if name is None:
                name = getattr(extension, 'default_name', None)
                if name is None:
                    name = getattr(extension, '__name__', None)
                    if name is None:
                        raise TypeError('name is not given for the extension')
        if name == 'training':
            raise ValueError(
                'the name "training" is prohibited as an extension name')

        if trigger is None:
            trigger = getattr(extension, 'trigger', (1, 'iteration'))
        trigger = trigger_module.get_trigger(trigger)

        if priority is None:
            priority = getattr(
                extension, 'priority', extension_module.PRIORITY_READER)

        if invoke_before_training is None:
            invoke_before_training = getattr(
                extension, 'invoke_before_training', False)

        modified_name = name
        ordinal = 0
        while modified_name in self._extensions:
            ordinal += 1
            modified_name = '%s_%d' % (name, ordinal)

        extension.name = modified_name
        self._extensions[modified_name] = _ExtensionEntry(
            extension, priority, trigger, invoke_before_training)

    def get_extension(self, name):
        """Returns the extension of a given name.

        Args:
            name (str): Name of the extension.

        Returns:
            Extension.

        """
        extensions = self._extensions
        if name in extensions:
            return extensions[name].extension
        else:
            raise ValueError('extension %s not found' % name)

    def run(self):
        """Executes the training loop.

        This method is the core of ``Trainer``. It executes the whole loop of
        training the models.

        Note that this method cannot run multiple times for one trainer object.

        """
        if self._done:
            raise RuntimeError('cannot run training loop multiple times')

        try:
            os.makedirs(self.out)
        except OSError:
            pass

        # sort extensions by priorities
        extension_order = sorted(
            self._extensions.keys(),
            key=lambda name: self._extensions[name].priority, reverse=True)
        extensions = [(name, self._extensions[name])
                      for name in extension_order]

        self._start_at = time.time()

        # invoke extensions before the loop
        for _, entry in extensions:
            if entry.invoke_before_training:
                entry.extension(self)

        update = self.updater.update
        reporter = self.reporter
        stop_trigger = self.stop_trigger

        # main training loop
        try:
            while not stop_trigger(self):
                self.observation = {}
                with reporter.scope(self.observation):
                    update()
                    for name, entry in extensions:
                        if entry.trigger(self):
                            entry.extension(self)
        finally:
            for _, entry in extensions:
                finalize = getattr(entry.extension, 'finalize', None)
                if finalize:
                    finalize()
            self.updater.finalize()

        self._final_elapsed_time = self.elapsed_time
        self._done = True

    def serialize(self, serializer):
        self.updater.serialize(serializer['updater'])
        if hasattr(self.stop_trigger, 'serialize'):
            self.stop_trigger.serialize(serializer['stop_trigger'])

        s = serializer['extensions']
        t = serializer['extension_triggers']
        for name, entry in six.iteritems(self._extensions):
            if hasattr(entry.extension, 'serialize'):
                entry.extension.serialize(s[name])
            if hasattr(entry.trigger, 'serialize'):
                entry.trigger.serialize(t[name])

        if isinstance(serializer, serializer_module.Serializer):
            serializer('_snapshot_elapsed_time', self.elapsed_time)
        else:
            self._snapshot_elapsed_time = serializer(
                '_snapshot_elapsed_time', 0.0)
