import os
import shutil
import tempfile

from chainer.serializers import npz
from chainer.training import extension


def snapshot(savefun=npz.save_npz,
             filename='snapshot_iter_{.updater.iteration}'):
    """Return a trainer extension to take snapshots of the trainer.

    This extension serializes the trainer object and saves it to the output
    directory. It is used to support resuming the training loop from the saved
    state.

    This extension is called once for each epoch by default.

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

    """
    @extension.make_extension(name='snapshot', trigger=(1, 'epoch'))
    def ext(trainer):
        fname = filename.format(trainer)
        fd, tmppath = tempfile.mkstemp(prefix=fname, dir=trainer.out)
        try:
            savefun(tmppath, trainer)
        finally:
            os.close(fd)
        os.rename(tmppath, os.path.join(trainer.out, fname))

    return ext
