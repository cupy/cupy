<div align="center"><img src="https://raw.githubusercontent.com/cupy/cupy/master/docs/image/cupy_logo_1000px.png" width="400"/></div>

# CuPy : NumPy & SciPy for GPU

[![pypi](https://img.shields.io/pypi/v/cupy.svg)](https://pypi.python.org/pypi/cupy)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/cupy.svg)](https://anaconda.org/conda-forge/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy.svg)](https://github.com/cupy/cupy)
[![coveralls](https://img.shields.io/coveralls/cupy/cupy.svg)](https://coveralls.io/github/cupy/cupy)
[![Gitter](https://badges.gitter.im/cupy/community.svg)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)

[**Website**](https://cupy.dev/)
| [**Install**](https://docs.cupy.dev/en/stable/install.html)
| [**Tutorial**](https://docs.cupy.dev/en/stable/user_guide/basic.html)
| [**Examples**](https://github.com/cupy/cupy/tree/master/examples)
| [**Documentation**](https://docs.cupy.dev/en/stable/)
| [**API Reference**](https://docs.cupy.dev/en/stable/reference/)
| [**Forum**](https://groups.google.com/forum/#!forum/cupy)

CuPy is a NumPy/SciPy-compatible array library for GPU-accelerated computing with Python.
CuPy acts as a [drop-in replacement](https://docs.cupy.dev/en/stable/reference/comparison.html) to run existing NumPy/SciPy code on NVIDIA CUDA or AMD ROCm platforms.

```py
>>> import cupy as cp
>>> x = cp.arange(6).reshape(2, 3).astype('f')
>>> x
array([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.]], dtype=float32)
>>> x.sum(axis=1)
array([  3.,  12.], dtype=float32)
```

CuPy also provides access to low-level CUDA features.
You can pass `ndarray` to existing CUDA C/C++ programs via [RawKernels](https://docs.cupy.dev/en/stable/user_guide/kernel.html#raw-kernels), use [Streams](https://docs.cupy.dev/en/stable/reference/cuda.html) for performance, or even call [CUDA Runtime APIs](https://docs.cupy.dev/en/stable/reference/cuda.html#runtime-api) directly.

## Installation

Wheels (precompiled binary packages) are available for Linux and Windows.
Choose the right package for your platform.

| Platform              | Architecture      | Command                                                       |
| --------------------- | ----------------- | ------------------------------------------------------------- |
| CUDA 10.2             | x86_64            | `pip install cupy-cuda102`                                    |
|                       | aarch64           | `pip install cupy-cuda102 -f https://pip.cupy.dev/aarch64`    |
| CUDA 11.0             | x86_64            | `pip install cupy-cuda110`                                    |
| CUDA 11.1             | x86_64            | `pip install cupy-cuda111`                                    |
| CUDA 11.2 or later    | x86_64            | `pip install cupy-cuda11x`                                    |
|                       | aarch64           | `pip install cupy-cuda11x -f https://pip.cupy.dev/aarch64`    |
| ROCm 4.3 (*)          | x86_64            | `pip install cupy-rocm-4-3`                                   |
| ROCm 5.0 (*)          | x86_64            | `pip install cupy-rocm-5-0`                                   |

(\*) ROCm support is an experimental feature. Refer to the [docs](https://docs.cupy.dev/en/latest/install.html#using-cupy-on-amd-gpu-experimental) for details.

Append `--pre -f https://pip.cupy.dev/pre` options to install pre-releases (e.g., `pip install cupy-cuda11x --pre -f https://pip.cupy.dev/pre`).
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
**CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations.**
*Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)*, (2017).
[[PDF](http://learningsys.org/nips17/assets/papers/paper_16.pdf)]

```bibtex
@inproceedings{cupy_learningsys2017,
  author       = "Okuta, Ryosuke and Unno, Yuya and Nishino, Daisuke and Hido, Shohei and Loomis, Crissman",
  title        = "CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations",
  booktitle    = "Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)",
  year         = "2017",
  url          = "http://learningsys.org/nips17/assets/papers/paper_16.pdf"
}
```
