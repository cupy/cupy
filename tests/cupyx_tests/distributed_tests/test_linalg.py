import dataclasses
import math

import pytest

import numpy
import cupy
from cupy import testing
from cupyx.distributed.array import _linalg

from cupyx.distributed import array
from cupyx.distributed.array import REPLICA, SUM


class ArrayConfig:
    size: int
    shape: tuple[int, ...]
    index_map: dict[int, list[tuple[slice, ...]]]

    def __init__(
        self, shape: tuple[int, ...],
        index_map: dict[int, list[tuple[slice, ...]]]
    ) -> None:
        self.size = math.prod(shape)
        self.shape = shape
        self.index_map = index_map

    def instantiate(
        self, mode: str = REPLICA,
    ) -> tuple[numpy.ndarray, array.DistributedArray]:
        np_arr = numpy.arange(self.size).reshape(self.shape)
        d_arr = array.distributed_array(np_arr, self.index_map, mode)
        return np_arr, d_arr


def make_1d_config(
    partitions: list[int],
    devices: list[set[int]],
) -> ArrayConfig:
    index_map: dict[int, list[tuple[slice, ...]]] = {}
    for i in range(len(partitions) - 1):
        idx = (slice(partitions[i], partitions[i+1]),)
        for dev in devices[i]:
            index_map.setdefault(dev, []).append(idx)

    shape = (partitions[-1],)
    return ArrayConfig(shape, index_map)


def make_2d_config(
    i_partitions: list[int],
    j_partitions: list[int],
    devices: list[list[set[int]]],
) -> ArrayConfig:
    shape = (i_partitions[-1], j_partitions[-1])
    index_map = _linalg.make_2d_index_map(i_partitions, j_partitions, devices)
    return ArrayConfig(shape, index_map)


@dataclasses.dataclass
class MatMulConfig:
    a: ArrayConfig
    b: ArrayConfig

    def instantiate(
        self, mode: str = REPLICA,
    ) -> tuple[numpy.ndarray, array.DistributedArray,
               numpy.ndarray, array.DistributedArray]:
        return self.a.instantiate(mode) + self.b.instantiate(mode)


def combine_configs(
    config_1: ArrayConfig, config_2: ArrayConfig,
) -> ArrayConfig:
    assert config_1.shape == config_2.shape
    shape = (3,) + config_1.shape

    index_map = {}
    for dev, idxs in config_1.index_map.items():
        index_map[dev] = [(slice(2),) + idx for idx in idxs]
    for dev, idxs in config_2.index_map.items():
        index_map[dev] += [(slice(2, 3),) + idx for idx in idxs]

    return ArrayConfig(shape, index_map)


config_1x2_2x2 = MatMulConfig(
    make_2d_config([0, 10], [0, 14, 20],
                   [[{0, 1}, {2, 3}]]),
    make_2d_config([0, 14, 20], [0, 8, 12],
                   [[{0}, {1}],
                    [{2}, {3}]]))


config_2x2_2x2 = MatMulConfig(
    make_2d_config([0, 6, 10], [0, 11, 20],
                   [[{0, 2}, {1, 3}],
                    [{2, 3}, {0, 1}]]),
    make_2d_config([0, 11, 20], [0, 7, 12],
                   [[{0, 2}, {2, 3}],
                    [{0, 1}, {1, 3}]]))


config_1x4_4x1 = MatMulConfig(
    make_2d_config([0, 10], [0, 3, 7, 12, 20],
                   [[{0}, {1}, {2}, {3}]]),
    make_2d_config([0, 3, 7, 12, 20], [0, 12],
                   [[{0}],
                    [{1}],
                    [{2}],
                    [{3}]]))


config_2x3_3x2 = MatMulConfig(
    make_2d_config([0, 2, 10], [0, 3, 7, 20],
                   [[{0, 1}, {1, 3}, {0, 2}],
                    [{0, 3}, {1, 2}, {2, 3}]]),
    make_2d_config([0, 3, 7, 20], [0, 4, 12],
                   [[{0}, {1, 3}],
                    [{1, 2}, {1, 3}],
                    [{0, 2}, {2, 3}]]))


configs_a = [
    make_1d_config([0, 11, 20],
                   [{0}, {1}]),
    make_2d_config([0, 6, 20], [0, 11, 20],
                   [[{0}, {1}],
                    [{0}, {1}]])]


configs_b = [
    make_1d_config([0, 11, 20],
                   [{0}, {1}]),
    make_2d_config([0, 11, 20], [0, 7, 20],
                   [[{0}, {0}],
                    [{1}, {1}]])]


@testing.multi_gpu(4)
class TestDistributedMatMul:
    @pytest.mark.parametrize(
        'config',
        [config_1x2_2x2, config_2x2_2x2, config_1x4_4x1, config_2x3_3x2])
    @pytest.mark.parametrize('mode', [REPLICA, SUM])
    def test_matmul(self, config, mode):
        np_a, d_a, np_b, d_b = config.instantiate(mode)
        np_c = np_a @ np_b
        d_c = d_a @ d_b
        testing.assert_array_equal(d_c.get(), np_c, strict=True)

    def test_incompatible_blockings(self):
        wrong_config = MatMulConfig(config_1x2_2x2.a, config_2x3_3x2.b)
        np_a, d_a, np_b, d_b = wrong_config.instantiate()
        with pytest.raises(RuntimeError, match=r'Inconsistent'):
            d_a @ d_b

    def test_high_dim(self):
        config_a = combine_configs(config_1x2_2x2.a, config_2x2_2x2.a)
        config_b = combine_configs(config_1x2_2x2.b, config_2x2_2x2.b)
        config = MatMulConfig(config_a, config_b)

        np_a, d_a, np_b, d_b = config.instantiate()
        np_c = np_a @ np_b
        d_c = d_a @ d_b

        testing.assert_array_equal(d_c.get(), np_c)

    @pytest.mark.parametrize('config_a', configs_a)
    @pytest.mark.parametrize('config_b', configs_b)
    def test_1d(self, config_a, config_b):
        np_a, d_a = config_a.instantiate()
        np_b, d_b = config_b.instantiate()
        np_c = np_a @ np_b
        d_c = d_a @ d_b
        testing.assert_array_equal(d_c.get(), np_c, strict=True)

    @pytest.mark.parametrize('config', [config_1x2_2x2])
    @pytest.mark.parametrize('mode', [REPLICA, SUM])
    def test_matmul_various_ops(self, config, mode):
        config = MatMulConfig(
            config.a,
            combine_configs(config.b, config.b))
        np_a, d_a, np_b, d_b = config.instantiate(mode)

        index_map_a = d_a.index_map
        index_map_a[0] = index_map_a[0] * 2

        np_a2 = np_a + 1
        d_a2 = d_a.reshard(index_map_a) + array.distributed_array(
            cupy.ones_like(d_a.get()), index_map_a)

        np_b2 = np_b.sum(axis=0)
        d_b2 = d_b.sum(axis=0)

        index_map_b = d_b2.index_map
        index_map_b[0], index_map_b[1] = index_map_b[1], index_map_b[0]
        d_b3 = d_b2.reshard(index_map_b)

        np_c = np_a2 @ np_b2
        d_c = d_a2 @ d_b3
        testing.assert_array_equal(d_c.get(), np_c, strict=True)
