"""Kernel cache backend infrastructure for CuPy compiler.

This module provides cache backend abstractions for storing compiled kernel
binaries. The cache helps avoid recompiling kernels that have already been
compiled.

.. warning::
   User-defined custom cache backends are experimental and should not expect
   API stability. The interface may change in future releases without
   deprecation warnings.
"""

from __future__ import annotations

import abc
import hashlib
import os
import tempfile


def _hash_hexdigest(value: bytes) -> str:
    """Compute SHA1 hash of the given bytes."""
    return hashlib.sha1(value, usedforsecurity=False).hexdigest()


_hash_length = len(_hash_hexdigest(b''))  # 40 for SHA1
_default_cache_dir = os.path.expanduser('~/.cupy/kernel_cache')


class KernelCacheBackend(abc.ABC):
    """Abstract base class for kernel cache storage backends.

    This class defines the interface for pluggable cache storage backends.
    Subclasses should implement methods to load and save compiled kernel
    binaries.

    .. warning::
       User-defined custom cache backends are experimental and should not
       expect API stability. The interface may change in future releases
       without deprecation warnings.
    """

    @abc.abstractmethod
    def load(self, name: str) -> bytes | None:
        """Load a cached kernel binary.

        Args:
            name (str): The cache key (filename) for the compiled kernel.

        Returns:
            bytes or None: The cubin binary data (without hash prefix) if
                found and valid, None otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, name: str, cubin: bytes, source: str) -> None:
        """Save a compiled kernel binary to cache.

        This method may perform I/O asynchronously to avoid blocking
        kernel execution.

        Args:
            name (str): The cache key (filename) for the compiled kernel.
            cubin (bytes): The compiled kernel binary data.
            source (str): The CUDA source code.
        """
        raise NotImplementedError


class DiskKernelCacheBackend(KernelCacheBackend):
    """Disk-based kernel cache storage backend.

    This backend stores compiled kernel binaries in a directory on disk.
    """

    def __init__(self, cache_dir: str | None = None):
        """Initialize the disk cache backend.

        Args:
            cache_dir (str, optional): Directory to store cache files.
                Defaults to CUPY_CACHE_DIR environment variable or
                ~/.cupy/kernel_cache.
        """
        if cache_dir is None:
            cache_dir = os.environ.get('CUPY_CACHE_DIR', _default_cache_dir)
        self._cache_dir = cache_dir
        if not os.path.isdir(self._cache_dir):
            os.makedirs(self._cache_dir, exist_ok=True)
        self._save_cuda_source = bool(
            os.environ.get('CUPY_CACHE_SAVE_CUDA_SOURCE'))

    def _encode_cubin(self, cubin: bytes) -> bytes:
        """Encode a cubin binary to the on-disk format (hash prefix + cubin).

        Args:
            cubin (bytes): Raw compiled kernel binary data.

        Returns:
            bytes: SHA-1 hash (ASCII hex) prepended to the cubin.
        """
        return _hash_hexdigest(cubin).encode('ascii') + cubin

    def _decode_cubin(self, data: bytes) -> bytes | None:
        """Decode and validate data in the on-disk format.

        Args:
            data (bytes): Raw bytes in the on-disk format (hash + cubin).

        Returns:
            bytes or None: The raw cubin if the hash is valid, None otherwise.
        """
        if len(data) < _hash_length:
            return None
        hash_stored = data[:_hash_length]
        cubin = data[_hash_length:]
        if hash_stored != _hash_hexdigest(cubin).encode('ascii'):
            return None
        return cubin

    def _write_encoded(self, name: str, data: bytes) -> None:
        """Atomically write pre-encoded (hash + cubin) data to the cache dir.

        Unlike :meth:`save`, this method accepts data that is already in the
        on-disk format (i.e. the SHA-1 hash prefix is already prepended).
        It does not save a `.cu` source file.

        Args:
            name (str): The cache key (filename) for the compiled kernel.
            data (bytes): Pre-encoded bytes (hash prefix + cubin).
        """
        path = os.path.join(self._cache_dir, name)
        with tempfile.NamedTemporaryFile(
                dir=self._cache_dir, delete=False) as tf:
            tf.write(data)
            temp_path = tf.name
        try:
            os.replace(temp_path, path)
        except PermissionError:
            pass  # Race on Windows; assume existing file is fine

    def load(self, name: str) -> bytes | None:
        """Load a cached kernel binary from disk.

        Args:
            name (str): The cache key (filename) for the compiled kernel.

        Returns:
            bytes or None: The cubin binary data (without hash prefix) if
                found and valid, None otherwise.
        """
        path = os.path.join(self._cache_dir, name)
        if not os.path.exists(path):
            return None

        with open(path, 'rb') as file:
            data = file.read()

        return self._decode_cubin(data)

    def save(self, name: str, cubin: bytes, source: str) -> None:
        """Save a compiled kernel binary to disk.

        Args:
            name (str): The cache key (filename) for the compiled kernel.
            cubin (bytes): The compiled kernel binary data.
            source (str): The CUDA source code.
        """
        self._write_encoded(name, self._encode_cubin(cubin))

        # Save .cu source file along with .cubin if requested
        if self._save_cuda_source:
            path = os.path.join(self._cache_dir, name)
            with open(path + '.cu', 'w') as f:
                f.write(source)
