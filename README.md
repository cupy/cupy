<div align="center"><img src="https://raw.githubusercontent.com/cupy/cupy/master/docs/image/cupy_logo_1000px.png" width="400"/></div>

# CuPy : A NumPy-compatible array library accelerated by CUDA

[![pypi](https://img.shields.io/pypi/v/cupy.svg)](https://pypi.python.org/pypi/cupy)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/cupy.svg)](https://anaconda.org/conda-forge/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy.svg)](https://github.com/cupy/cupy)
[![coveralls](https://img.shields.io/coveralls/cupy/cupy.svg)](https://coveralls.io/github/cupy/cupy)
[![Gitter](https://badges.gitter.im/cupy/community.svg)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)

[**Website**](https://cupy.dev/)
| [**Docs**](https://docs.cupy.dev/en/stable/)
| [**Install Guide**](https://docs.cupy.dev/en/stable/install.html)
| [**Tutorial**](https://docs.cupy.dev/en/stable/tutorial/)
| [**Examples**](https://github.com/cupy/cupy/tree/master/examples)
| [**API Reference**](https://docs.cupy.dev/en/stable/reference/)
| [**Forum**](https://groups.google.com/forum/#!forum/cupy)

*CuPy* is an implementation of NumPy-compatible multi-dimensional array on CUDA.
CuPy consists of the core multi-dimensional array class, `cupy.ndarray`, and [many functions](https://docs.cupy.dev/en/stable/reference/comparison.html) on it.

## Installation

Wheels (precompiled binary packages) are available for Linux (x86_64) and Windows (amd64).
Choose the right package for your platform.

| Platform  | Command                        |
| --------- | ------------------------------ |
| CUDA 9.0  | `pip install cupy-cuda90`      |
| CUDA 9.2  | `pip install cupy-cuda92`      |
| CUDA 10.0 | `pip install cupy-cuda100`     |
| CUDA 10.1 | `pip install cupy-cuda101`     |
| CUDA 10.2 | `pip install cupy-cuda102`     |
| CUDA 11.0 | `pip install cupy-cuda110`     |
| CUDA 11.1 | `pip install cupy-cuda111`     |
| CUDA 11.2 | `pip install cupy-cuda112`     |
| ROCm 4.0  | `pip install cupy-rocm-4-0` (experimental; see [docs](https://docs.cupy.dev/en/latest/install.html#using-cupy-on-amd-gpu-experimental) for details) |

See the [Installation Guide](https://docs.cupy.dev/en/stable/install.html) if you are using Conda/Anaconda or building from source.

## Run on Docker

Use [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to run CuPy image with GPU.

```
$ docker run --gpus all -it cupy/cupy
```

## More information

- [Release Notes](https://github.com/cupy/cupy/releases)
- [Projects using CuPy](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)
- [Contribution Guide](https://docs.cupy.dev/en/stable/contribution.html)

## License

MIT License (see `LICENSE` file).

CuPy is designed based on NumPy's API and SciPy's API (see `docs/LICENSE_THIRD_PARTY` file).

CuPy is being maintained and developed by [Preferred Networks Inc.](https://preferred.jp/en/) and [community contributors](https://github.com/cupy/cupy/graphs/contributors).

## Reference

Ryosuke Okuta, Yuya Unno, Daisuke Nishino, Shohei Hido and Crissman Loomis.
CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations.
*Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)*, (2017).
[URL](http://learningsys.org/nips17/assets/papers/paper_16.pdf)

```
@inproceedings{cupy_learningsys2017,
  author       = "Okuta, Ryosuke and Unno, Yuya and Nishino, Daisuke and Hido, Shohei and Loomis, Crissman",
  title        = "CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations",
  booktitle    = "Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)",
  year         = "2017",
  url          = "http://learningsys.org/nips17/assets/papers/paper_16.pdf"
}
```
