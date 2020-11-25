<div align="center"><img src="https://raw.githubusercontent.com/cupy/cupy/master/docs/image/cupy_logo_1000px.png" width="400"/></div>

# CuPy : A NumPy-compatible array library accelerated by CUDA

[![pypi](https://img.shields.io/pypi/v/cupy.svg)](https://pypi.python.org/pypi/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy.svg)](https://github.com/cupy/cupy)
[![travis](https://img.shields.io/travis/cupy/cupy.svg)](https://travis-ci.org/cupy/cupy)
[![coveralls](https://img.shields.io/coveralls/cupy/cupy.svg)](https://coveralls.io/github/cupy/cupy)
[![Gitter](https://badges.gitter.im/cupy/community.svg)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)

[**Website**](https://cupy.dev/)
| [**Docs**](https://docs.cupy.dev/en/stable/)
| [**Install Guide**](https://docs.cupy.dev/en/stable/install.html)
| [**Tutorial**](https://docs.cupy.dev/en/stable/tutorial/)
| [**Examples**](https://github.com/cupy/cupy/tree/master/examples)
| [**API Reference**](https://docs.cupy.dev/en/stable/reference/)
| **Forum** ([en](https://groups.google.com/forum/#!forum/cupy), [ja](https://groups.google.com/forum/#!forum/cupy-ja))

*CuPy* is an implementation of NumPy-compatible multi-dimensional array on CUDA.
CuPy consists of the core multi-dimensional array class, `cupy.ndarray`, and [many functions](https://docs.cupy.dev/en/stable/reference/comparison.html) on it.

## Installation

Wheels (precompiled binary packages) are available for Linux (Python 3.5+) and Windows (Python 3.6+).
Choose the right package for your CUDA Toolkit version.

| CUDA  | Command                        |
| ----- | ------------------------------ |
| v9.0  | `pip install cupy-cuda90`      |
| v9.2  | `pip install cupy-cuda92`      |
| v10.0 | `pip install cupy-cuda100`     |
| v10.1 | `pip install cupy-cuda101`     |
| v10.2 | `pip install cupy-cuda102`     |
| v11.0 | `pip install cupy-cuda110`     |
| v11.1 | `pip install cupy-cuda111`     |

See the [Installation Guide](https://docs.cupy.dev/en/stable/install.html) if you are using Conda/Anaconda or to build from source.

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
