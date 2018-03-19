<div align="center"><img src="docs/image/cupy_logo_1000px.png" width="400"/></div>

# CuPy : NumPy-like API accelerated with CUDA

[![pypi](https://img.shields.io/pypi/v/cupy.svg)](https://pypi.python.org/pypi/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy.svg)](https://github.com/cupy/cupy)
[![travis](https://img.shields.io/travis/cupy/cupy.svg)](https://travis-ci.org/cupy/cupy)
[![coveralls](https://img.shields.io/coveralls/cupy/cupy.svg)](https://coveralls.io/github/cupy/cupy)
[![Read the Docs](https://readthedocs.org/projects/cupy/badge/?version=stable)](https://docs-cupy.chainer.org/en/stable/)

[**Website**](https://cupy.chainer.org/)
| [**Docs**](https://docs-cupy.chainer.org/en/stable/)
| [**Install Guide**](https://docs-cupy.chainer.org/en/stable/install.html)
| [**Tutorial**](https://docs-cupy.chainer.org/en/stable/tutorial/)
| **Examples** ([Official](https://github.com/cupy/cupy/tree/master/examples))
| **Forum** ([en](https://groups.google.com/forum/#!forum/cupy), [ja](https://groups.google.com/forum/#!forum/cupy-ja))

*CuPy* is an implementation of NumPy-compatible multi-dimensional array on CUDA.
CuPy consists of the core multi-dimensional array class, `cupy.ndarray`, and many functions on it.
It supports a subset of `numpy.ndarray` interface.

## Installation

For detailed instructions on installing CuPy, see [the installation guide](https://docs-cupy.chainer.org/en/stable/install.html).

You can install CuPy using `pip`:

```sh
$ pip install cupy
```

Note that if you want to enable CUDA, cuDNN, and/or NCCL, they need to be set up before installation of CuPy.

## Run with Docker

We provide the official Docker image.
Use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) command to run CuPy image with GPU.
You can login to the environment with bash, and run the Python interpreter.

```
$ nvidia-docker run -it cupy/cupy /bin/bash
```

## Development

Please see [the contribution guide](https://docs-cupy.chainer.org/en/stable/contribution.html).

## More information

- [Release notes](https://github.com/cupy/cupy/releases)
- [Projects using CuPy](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)

## License

MIT License (see `LICENSE` file).


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
