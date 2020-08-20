<div align="center"><img src="https://raw.githubusercontent.com/cupy/cupy/master/docs/image/cupy_logo_1000px.png" width="400"/></div>

# CuPy : NumPy-like API accelerated with CUDA

[![pypi](https://img.shields.io/pypi/v/cupy.svg)](https://pypi.python.org/pypi/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy.svg)](https://github.com/cupy/cupy)
[![travis](https://img.shields.io/travis/cupy/cupy.svg)](https://travis-ci.org/cupy/cupy)
[![coveralls](https://img.shields.io/coveralls/cupy/cupy.svg)](https://coveralls.io/github/cupy/cupy)
[![Read the Docs](https://readthedocs.org/projects/cupy/badge/?version=stable)](https://docs.cupy.dev/en/stable/)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)

[**Website**](https://cupy.dev/)
| [**Docs**](https://docs.cupy.dev/en/stable/)
| [**Install Guide**](https://docs.cupy.dev/en/stable/install.html)
| [**Tutorial**](https://docs.cupy.dev/en/stable/tutorial/)
| [**Examples**](https://github.com/cupy/cupy/tree/master/examples)
| **Forum** ([en](https://groups.google.com/forum/#!forum/cupy), [ja](https://groups.google.com/forum/#!forum/cupy-ja))

*CuPy* is an implementation of NumPy-compatible multi-dimensional array on CUDA.
CuPy consists of the core multi-dimensional array class, `cupy.ndarray`, and many functions on it.
It supports a subset of `numpy.ndarray` interface.

## Installation

For detailed instructions on installing CuPy, see [the installation guide](https://docs.cupy.dev/en/stable/install.html).

You can install CuPy using `pip`:

```sh
(Binary Package for CUDA 9.0)
$ pip install cupy-cuda90

(Binary Package for CUDA 9.2)
$ pip install cupy-cuda92

(Binary Package for CUDA 10.0)
$ pip install cupy-cuda100

(Binary Package for CUDA 10.1)
$ pip install cupy-cuda101

(Binary Package for CUDA 10.2)
$ pip install cupy-cuda102

(Binary Package for CUDA 11.0)
$ pip install cupy-cuda110

(Source Package)
$ pip install cupy
```

The latest version of cuDNN and NCCL libraries are included in binary packages (wheels).
For the source package, you will need to install cuDNN/NCCL before installing CuPy, if you want to use it.

## Run with Docker

We provide the official Docker image.
Use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) command to run CuPy image with GPU.
You can login to the environment with bash, and run the Python interpreter.

```
$ nvidia-docker run -it cupy/cupy /bin/bash
```

## Development

Please see [the contribution guide](https://docs.cupy.dev/en/stable/contribution.html).

## More information

- [Release notes](https://github.com/cupy/cupy/releases)
- [Projects using CuPy](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)

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
