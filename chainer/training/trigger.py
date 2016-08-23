from chainer.training import triggers
from chainer.training.triggers import interval


IntervalTrigger = interval.IntervalTrigger
_never_fire_trigger = triggers._never_fire_trigger
get_trigger = triggers.get_trigger
