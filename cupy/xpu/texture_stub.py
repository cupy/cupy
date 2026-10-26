"""Pure-Python stub for :mod:`cupy.cuda.texture` on non-CUDA backends.

The real implementation (``cupy/cuda/texture.pyx``) is CUDA-only: it talks to
the CUDA runtime texture-object API, which has no equivalent on Ascend NPU.
However, ``cupyx.scipy.ndimage`` (via ``cupyx._texture``) and downstream
projects such as ``scipy``'s array-API mode import ``cupy.cuda.texture`` at
module load time even when the texture-memory code path is never exercised.

This module mirrors the class names of ``cupy/cuda/texture.pxd`` so that
``from cupy.cuda import texture`` succeeds on Ascend. All constructors raise
``NotImplementedError`` — importing is safe, but actually *using* texture
objects is not supported on this backend.

See also ``cupy/cuda/__init__.py``, which registers ``cupy.cuda.texture`` as
a ``sys.modules`` alias pointing to this module when the Ascend backend is
active.
"""

__all__ = [
    'ChannelFormatDescriptor',
    'ResourceDescriptor',
    'TextureDescriptor',
    'CUDAarray',
    'TextureObject',
    'SurfaceObject',
]


class ChannelFormatDescriptor:
    """Stub for :class:`cupy.cuda.texture.ChannelFormatDescriptor`.

    Equivalent to ``cudaChannelFormatDesc``.
    """

    def __init__(self, x, y, z, w, f):
        raise NotImplementedError(
            'texture memory is not supported on this backend '
            '(no CUDA texture API)')

    def get_channel_format(self):
        raise NotImplementedError(
            'texture memory is not supported on this backend '
            '(no CUDA texture API)')


class ResourceDescriptor:
    """Stub for :class:`cupy.cuda.texture.ResourceDescriptor`.

    Equivalent to ``cudaResourceDesc``.
    """

    def __init__(self, restype, cuArr=None, arr=None, chDesc=None,
                 sizeInBytes=0, width=0, height=0, pitchInBytes=0):
        raise NotImplementedError(
            'texture memory is not supported on this backend '
            '(no CUDA texture API)')


class TextureDescriptor:
    """Stub for :class:`cupy.cuda.texture.TextureDescriptor`.

    Equivalent to ``cudaTextureDesc``.
    """

    def __init__(self, addressModes=None, filterMode=0, readMode=0,
                 sRGB=None, borderColors=None, normalizedCoords=None,
                 maxAnisotropy=None):
        raise NotImplementedError(
            'texture memory is not supported on this backend '
            '(no CUDA texture API)')


class CUDAarray:
    """Stub for :class:`cupy.cuda.texture.CUDAarray`.

    Equivalent to ``cudaArray``.
    """

    def __init__(self, desc, width, height=0, depth=0, flags=0):
        raise NotImplementedError(
            'texture memory is not supported on this backend '
            '(no CUDA texture API)')

    def copy_from(self, data, stream=None):
        raise NotImplementedError(
            'texture memory is not supported on this backend '
            '(no CUDA texture API)')

    def copy_to(self, data, stream=None):
        raise NotImplementedError(
            'texture memory is not supported on this backend '
            '(no CUDA texture API)')


class TextureObject:
    """Stub for :class:`cupy.cuda.texture.TextureObject`.

    Equivalent to ``cudaTextureObject_t``.
    """

    def __init__(self, ResDesc, TexDesc):
        raise NotImplementedError(
            'texture memory is not supported on this backend '
            '(no CUDA texture API)')


class SurfaceObject:
    """Stub for :class:`cupy.cuda.texture.SurfaceObject`.

    Equivalent to ``cudaSurfaceObject_t``.
    """

    def __init__(self, ResDesc):
        raise NotImplementedError(
            'texture memory is not supported on this backend '
            '(no CUDA texture API)')
