from .util import *


print(__file__.split()[-1])


length = 3000


size = length * length
shape_a = (length, length)
shape_b = (length, length)


configs = assign_devices([
    None,
    MatMulConfig(
        make_2d_config([0, length], [0, length],
                       [[{0}]]),
        make_2d_config([0, length], [0, length],
                       [[{0}]])),
    MatMulConfig(
        make_2d_config([0, length], [0, length],
                       [[{0, 1}]]),
        make_2d_config([0, length], [0, length // 2, length],
                       [[{0}, {1}]])),
    MatMulConfig(
        make_2d_config([0, length * 2 // 3, length], [0, length],
                       [[{0, 1}],
                        [{2}]]),
        make_2d_config([0, length], [0, length // 2, length],
                       [[{0, 2}, {1, 2}]])),
    # MatMulConfig(
    #     make_2d_config([0, length // 17 * 12, length],
    #                    [0, length // 25 * 21, length],
    #                    [[{0, 1}, {1}],
    #                     [{2}, {3}]]),
    #     make_2d_config([0, length // 25 * 21, length],
    #                    [0, length // 126 * 75, length],
    #                    [[{0, 2}, {1, 2}],
    #                     [{1, 3}, {1, 3}]])),
    MatMulConfig(
        make_2d_config([0, length // 2, length], [0, length],
                       [[{0, 1}],
                        [{2, 3}]]),
        make_2d_config([0, length], [0, length // 2, length],
                       [[{0, 2}, {1, 3}]])),
])


incompatible_configs = assign_devices([
    None,
    MatMulConfig(
        make_2d_config([0, length], [0, length],
                       [[{0}]]),
        make_2d_config([0, length], [0, length],
                       [[{0}]])),
    MatMulConfig(
        make_2d_config([0, length], [0, length],
                       [[{0}]]),
        make_2d_config([0, length], [0, length // 2, length],
                       [[{0}, {1}]])),
    MatMulConfig(
        make_2d_config([0, length * 2 // 3, length], [0, length],
                       [[{0}],
                        [{2}]]),
        make_2d_config([0, length], [0, length // 2, length],
                       [[{1}, {2}]])),
    # MatMulConfig(
    #     make_2d_config([0, length // 17 * 12, length],
    #                    [0, length // 25 * 21, length],
    #                    [[{0, 1}, {1}],
    #                     [{2}, {3}]]),
    #     make_2d_config([0, length // 25 * 21, length],
    #                    [0, length // 126 * 75, length],
    #                    [[{0, 2}, {1, 2}],
    #                     [{1, 3}, {1, 3}]])),
    MatMulConfig(
        make_2d_config([0, length // 2, length], [0, length],
                       [[{0}],
                        [{1}]]),
        make_2d_config([0, length], [0, length // 2, length],
                       [[{2}, {3}]])),
])


def non_distributed():
    print('non-distributed')
    cp_a = cupy.arange(size).reshape(shape_a)
    cp_b = cupy.arange(size).reshape(shape_b)
    bench(lambda: cp_a @ cp_b, 1)


def distributed(n_dev=4):
    if n_dev == 0:
        return non_distributed()
    print(f'distributed ({n_dev=})')
    _, d_a, _, d_b = configs[n_dev].instantiate()
    bench(lambda: d_a @ d_b, n_dev)


def distributed_reshard(n_dev=4):
    if n_dev == 0:
        return non_distributed()

    print(f'distributed, reshard ({n_dev=})')

    _, d_a, _, d_b = incompatible_configs[n_dev].instantiate()

    index_map_a = configs[n_dev].a.index_map
    index_map_b = configs[n_dev].b.index_map
    d_a = d_a.reshard(index_map_a)
    d_a.wait_all_transfer()

    bench(lambda: d_a @ d_b.reshard(index_map_b), n_dev)


# def high_dim(n_dev=4):
#     print(f'high dim ({n_dev=})')

#     from lintest import ArrayConfig

#     def combine(
#             config_1: ArrayConfig, config_2: ArrayConfig) -> ArrayConfig:
#         assert config_1.shape == config_2.shape
#         shape = (3,) + config_1.shape

#         index_map = {}
#         for dev, idxs in config_1.index_map.items():
#             index_map[dev] = [(slice(2),) + idx for idx in idxs]
#         for dev, idxs in config_2.index_map.items():
#             index_map[dev] += [(slice(2, 3),) + idx for idx in idxs]

#         return ArrayConfig(shape, index_map)

#     config_a_2d = configs[n_dev].a
#     config_b_2d = configs[n_dev].b

#     config_a_3d = combine(config_a_2d, config_a_2d)
#     config_b_3d = combine(config_b_2d, config_b_2d)

#     np_a, d_a = config_a_3d.instantiate()
#     np_b, d_b = config_b_3d.instantiate()

#     bench(lambda: d_a @ d_b, n_dev)


non_distributed()
for n_dev in range(1, 5):
    distributed(n_dev)
for n_dev in range(1, 5):
    distributed_reshard(n_dev)


print('default stream')
for dev in devices:
    with cupy.cuda.Device(dev):
        streams[dev].__exit__()


for n_dev in range(1, 5):
    distributed_reshard(n_dev)
print()


for dev in devices:
    with cupy.cuda.Device(dev):
        streams[dev].__enter__()


# bench = repeat
# distributed_reshard(4)
# high_dim(4)

# non-distributed
# <lambda>  : CPU:  160.832 us  GPU-0: 553081.741 us
# distributed(n_dev=1)
# <lambda>  : CPU: 1374.556 us  GPU-0: 572821.344 us
# distributed(n_dev=2)
# <lambda>  : CPU: 1315.479 us  GPU-0: 286361.314 us  GPU-1: 284685.382 us
# distributed(n_dev=3)
# <lambda>  : CPU: 1367.968 us  GPU-0: 209571.167 us  GPU-1: 208731.445 us  GPU-2: 215310.767 us
# distributed(n_dev=4)
# <lambda>  : CPU: 1396.898 us  GPU-0: 200970.203 us  GPU-1: 199473.770 us  GPU-2: 197433.734 us  GPU-3: 195650.284 us

