import os
import shutil
import tempfile

from chainer.serializers import npz
from chainer.training import extension


def snapshot_object(target, filename, savefun=npz.save_npz,
                    trigger=(1, 'epoch')):
    """Returns a trainer extension to take snapshots of a given object.

    This extension serializes the given object and saves it to the output
    directory.

    This extension is called once for each epoch by default. The default
    priority is -100, which is lower than that of most built-in extensions.

    Args:
        target: Object to serialize.
        filename (str): Name of the file into which the object is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.
        savefun: Function to save the object. It takes two arguments: the
            output file path and the object to serialize.
        trigger: Trigger that decides when to take snapshot. It can be either
            an already built trigger object (i.e., a callable object that
            accepts a trainer object and returns a bool value), or a tuple in
            the form ``<int>, 'epoch'`` or ``<int>, 'iteration'``. In latter
            case, the tuple is passed to IntervalTrigger.

    Returns:
        An extension function.

    """
    @extension.make_extension(trigger=trigger, priority=-100)
    def snapshot_object(trainer):
        _snapshot_object(trainer, target, filename.format(trainer), savefun)

    return snapshot_object


def snapshot(savefun=npz.save_npz,
             filename='snapshot_iter_{.updater.iteration}',
             trigger=(1, 'epoch')):
    """Returns a trainer extension to take snapshots of the trainer.

    This extension serializes the trainer object and saves it to the output
    directory. It is used to support resuming the training loop from the saved
    state.

    This extension is called once for each epoch by default. The default
    priority is -100, which is lower than that of most built-in extensions.

    .. note::
       This extension first writes the serialized object to a temporary file
       and then rename it to the target file name. Thus, if the program stops
       right before the renaming, the temporary file might be left in the
       output directory.

    Args:
        savefun: Function to save the trainer. It takes two arguments: the
            output file path and the trainer object.
        filename (str): Name of the file into which the trainer is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method.
        trigger: Trigger that decides when to take snapshot. It can be either
            an already built trigger object (i.e., a callable object that
            accepts a trainer object and returns a bool value), or a tuple in
            the form ``<int>, 'epoch'`` or ``<int>, 'iteration'``. In latter
            case, the tuple is passed to IntervalTrigger.

    """
    @extension.make_extension(trigger=trigger, priority=-100)
    def snapshot(trainer):
        _snapshot_object(trainer, trainer, filename.format(trainer), savefun)

    return snapshot


def _snapshot_object(trainer, target, filename, savefun):
    fn = filename.format(trainer)
    prefix = 'tmp' + fn
    fd, tmppath = tempfile.mkstemp(prefix=prefix, dir=trainer.out)
    try:
        savefun(tmppath, target)
    except Exception:
        os.close(fd)
        os.remove(tmppath)
        raise
    os.close(fd)
    shutil.move(tmppath, os.path.join(trainer.out, fn))
