from chainer.trainer import extension
from chainer.trainer import trainer
from chainer.trainer import trigger
from chainer.trainer import updater


Extension = extension.Extension
make_extension = extension.make_extension

Trainer = trainer.Trainer

IntervalTrigger = trigger.IntervalTrigger
get_trigger = trigger.get_trigger

Updater = updater.Updater
StandardUpdater = updater.StandardUpdater
