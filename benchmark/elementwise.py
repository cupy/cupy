from .util import *


print(__file__.split()[-1])


length = 20000
size = length * length
shape = (length, length)
offset = 200


index_map = assign_devices([
    None,
    {0: slice(None)},
    {0: slice(length//2),
     1: slice(length//2, None)},
    {0: slice(length//3),
     1: slice(length//3, length * 2//3),
     2: slice(length * 2//3, None)},
    {0: slice(length//4),
     1: slice(length//4, length//2),
     2: slice(length//2, length * 3//4),
     3: slice(length * 3//4, None)},
])


index_map_2 = assign_devices([
    None,
    {0: slice(None)},
    {0: slice(length//2 - offset),
     1: slice(length//2 - offset, None)},
    {0: slice(length//3 - offset//2),
     1: slice(length//3 - offset//2, length * 2//3 - offset//2),
     2: slice(length * 2//3 - offset//2, None)},
    {0: slice(length//4 - offset//3),
     1: slice(length//4 - offset//3, length//2 - offset//3),
     2: slice(length//2 - offset//3, length * 3//4 - offset//3),
     3: slice(length * 3//4 - offset//3, None),}
])


index_map_3 = assign_devices([
    None,
    {0: (slice(None), slice(None))},
    {0: (slice(None), slice(length//2)),
     1: (slice(None), slice(length//2, None))},
    {0: (slice(None), slice(length//3)),
     1: (slice(None), slice(length//3, length * 2//3)),
     2: (slice(None), slice(length * 2//3, None))},
    {0: (slice(None), slice(length//4)),
     1: (slice(None), slice(length//4, length//2)),
     2: (slice(None), slice(length//2, length * 3//4)),
     3: (slice(None), slice(length * 3//4, None))},
])


def non_distributed():
    print('non-distributed')

    a = cupy.arange(size).reshape(shape)
    b = cupy.arange(size).reshape(shape)

    bench(lambda: cupy.add(a, b), 1)


def without_reshard(n_dev=4):
    print(f'distributed, no resharding ({n_dev=})')

    data = cupy.arange(size).reshape(shape)

    mapping = index_map[n_dev]

    a = distributed_array(data, mapping)
    b = distributed_array(data, mapping)

    bench(lambda: cupy.add(a, b), n_dev)


def with_reshard(n_dev=4):
    print(f'distributed, resharding ({n_dev=})')

    data = cupy.arange(size).reshape(shape)

    index_map_a = index_map[n_dev]
    index_map_b = index_map_2[n_dev]
    # index_map_b = index_map_3[n_dev]


    a = distributed_array(data, index_map_a)
    b = distributed_array(data, index_map_b)

    # assert_array_equal((a + b.reshard(index_map_a)).asnumpy(), data * 2)

    bench(lambda: cupy.add(a, b.reshard(index_map_a)), n_dev)


non_distributed()
for n_dev in range(1, 5):
    without_reshard(n_dev)
for n_dev in range(1, 5):
    with_reshard(n_dev)


print('default stream')
for dev in devices:
    with cupy.cuda.Device(dev):
        streams[dev].__exit__()


non_distributed()
for n_dev in range(1, 5):
    without_reshard(n_dev)
for n_dev in range(1, 5):
    with_reshard(n_dev)


for dev in devices:
    with cupy.cuda.Device(dev):
        streams[dev].__enter__()

