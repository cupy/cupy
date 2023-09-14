from .util import *


print(__file__.split()[-1])


length = 30000
size = length * length
shape = (length, length)
offset = 300


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


# def peer_access(n_dev=2):
#     data = cupy.arange(size).reshape(shape)

#     index_map_a = index_map[n_dev]
#     index_map_b = index_map_2[n_dev]

#     a = distributed_array(data, index_map_a)
#     b = distributed_array(data, index_map_b)

#     bench(lambda: cupy.add(a, b), n_dev)


non_distributed()
for n_dev in range(1, 5):
    without_reshard(n_dev)
for n_dev in range(1, 5):
    with_reshard(n_dev)

# bench = repeat
# non_distributed()
# without_reshard(1)
# with_reshard(4)

# bench = repeat
# without_reshard(4)
# with_reshard(4)


# non_distributed
# <lambda>    :  CPU:   28.664 us    GPU-0:   4674.621 us
# distributed w/o resharding
# <lambda>    :  CPU:   49.449 us    GPU-0:   4781.670 us
# <lambda>    :  CPU:   65.232 us    GPU-0:   2400.031 us    GPU-1:   2362.061 us
# <lambda>    :  CPU:   89.761 us    GPU-0:   1860.874 us    GPU-1:   1885.757 us    GPU-2:   1830.943 us
# <lambda>    :  CPU:  107.593 us    GPU-0:   1619.753 us    GPU-1:   1643.807 us    GPU-2:   1597.235 us    GPU-3:   1641.900 us
# distributed w/ resharding
# <lambda>    :  CPU:  131.774 us    GPU-0:   8041.175 us
# <lambda>    :  CPU:  405.713 us    GPU-0: 110798.029 us    GPU-1: 112345.754 us
# <lambda>    :  CPU: 4874.506 us    GPU-0: 122789.171 us    GPU-1: 140067.769 us    GPU-2: 140767.151 us
# <lambda>    :  CPU:    6696.417 us    GPU-0: 128559.319 us    GPU-1: 137364.449 us    GPU-2: 142683.874 us    GPU-3: 143071.862 us

# via host memory
# <lambda>    :  CPU: 720738.263 us    GPU-0: 722982.410 us    GPU-1: 766718.677 us
# <lambda>    :  CPU: 906939.640 us    GPU-0: 908613.531 us    GPU-1: 908655.511 us    GPU-2: 923855.664 us
# <lambda>    :  CPU: 1009633.326 us    GPU-0: 1011033.807 us    GPU-1: 1011078.857 us    GPU-2: 1011098.315 us    GPU-3: 1015491.211 us

# distributed, peer access
# <lambda>    :  CPU: 375.140 us   GPU-0: 76712.756 us   GPU-1: 76828.877 us


