from .util import *


print(__file__.split()[-1])


length = 70000
size = length * length // 2
shape = (length, length // 2)
overlap = 700


index_map_overlaps = assign_devices([
    None,
    {0: slice(None)},
    {0: slice(length // 2 + overlap),
     1: slice(length // 2, None)},
    {0: slice(length // 3 + overlap // 2),
     1: slice(length // 3, length * 2 // 3 + overlap // 2),
     2: slice(length * 2 // 3, None)},
    {0: slice(length // 4 + overlap // 3),
     1: slice(length // 4, length // 2 + overlap // 3),
     2: slice(length // 2, length * 3 // 4 + overlap // 3),
     3: slice(length * 3 // 4, None)},
])


def non_distributed():
    print('non-distributed')

    cp_a = cupy.arange(size).reshape(shape)
    bench(lambda: cp_a.sum(0), 1)


def no_mode_change(n_dev=4):
    if n_dev == 0:
        return non_distributed()

    print(f'no mode change ({n_dev=})')

    cp_a = cupy.arange(size).reshape(shape)
    d_a = distributed_array(
        cp_a, index_map_overlaps[n_dev], 'sum', comms)

    bench(lambda: d_a.sum(0), n_dev)


def mode_change(n_dev=4):
    if n_dev == 0:
        return non_distributed()

    print(f'mode change ({n_dev=})')
    cp_a = cupy.arange(size).reshape(shape)
    d_a = distributed_array(
        cp_a, index_map_overlaps[n_dev], 'replica', comms)

    bench(lambda: d_a.sum(0), n_dev)


non_distributed()
for n_dev in range(1, 5):
    no_mode_change(n_dev)
for n_dev in range(1, 5):
    mode_change(n_dev)

# bench = repeat
# mode_change(4)
