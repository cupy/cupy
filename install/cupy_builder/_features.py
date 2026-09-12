"""Build feature aggregation entry point.

The per-backend feature definitions now live in ``cupy_builder.features``:

* :mod:`cupy_builder.features.cuda`   - NVIDIA CUDA
* :mod:`cupy_builder.features.rocm`   - AMD ROCm / HIP
* :mod:`cupy_builder.features.ascend` - Huawei Ascend / CANN

This module only dispatches on the build context and returns the feature
dict consumed by ``cupy_builder.Context``.
"""

from __future__ import annotations

from cupy_builder import Context
from cupy_builder.features._base import Feature
from cupy_builder.features._base import from_dict as _from_dict
from cupy_builder.features.ascend import CUPY_ascend
from cupy_builder.features.cuda import CUDA_cuda
from cupy_builder.features.cuda import cuda_feature_dicts
from cupy_builder.features.cuda import CUDA_FEATURES_USED
from cupy_builder.features.cuda import get_cudart_static_libs
from cupy_builder.features.rocm import rocm_feature_dicts
from cupy_builder.features.rocm import ROCM_FEATURE_ORDER


__all__ = ['Feature', 'get_features']


def get_features(ctx: Context) -> dict[str, Feature]:
    if ctx.use_hip:
        dicts = rocm_feature_dicts()
        features = [_from_dict(dicts[name], ctx) for name in ROCM_FEATURE_ORDER]
        # COMMON_dlpack is shared by ROCm and CUDA
        features.append(_from_dict(_common_dlpack(), ctx))
    elif ctx.use_ascend:
        features = [
            CUPY_ascend(ctx),
            # _from_dict(_common_dlpack(), ctx)
        ]
    else:
        dicts = cuda_feature_dicts(get_cudart_static_libs())
        features = [CUDA_cuda(ctx)]
        features += [_from_dict(dicts[name], ctx) for name in CUDA_FEATURES_USED]
    return {f.name: f for f in features}


def _common_dlpack() -> dict:
    """The DLPack feature is backend independent."""
    return cuda_feature_dicts(get_cudart_static_libs())['COMMON_dlpack']
