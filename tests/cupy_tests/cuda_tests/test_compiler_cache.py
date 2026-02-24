from __future__ import annotations

import os
import tempfile

from cupy.cuda._compiler_cache import (
    DiskKernelCacheBackend,
    _hash_length,
    _default_cache_dir,
)


class TestDiskKernelCacheBackend:
    """Tests for DiskKernelCacheBackend implementation."""

    def test_init_cache_dir(self):
        """Test initialization with default cache directory."""
        backend = DiskKernelCacheBackend()
        cupy_cache_dir = os.environ.get('CUPY_CACHE_DIR')
        if cupy_cache_dir is None:
            assert backend._cache_dir == _default_cache_dir
        else:
            assert backend._cache_dir == cupy_cache_dir
        assert os.path.isdir(backend._cache_dir)

    def test_init_custom_cache_dir(self):
        """Test initialization with custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, 'custom_cache')
            backend = DiskKernelCacheBackend(cache_dir=cache_dir)
            assert backend._cache_dir == cache_dir
            assert os.path.isdir(cache_dir)

    def test_save_and_load(self):
        """Test basic save and load operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            name = 'test_kernel.cubin'
            cubin = b'compiled_kernel_binary'
            source = 'extern "C" __global__ void test() {}'

            # Save the kernel
            backend.save(name, cubin, source)

            # Load it back
            loaded_cubin = backend.load(name)
            assert loaded_cubin == cubin

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            result = backend.load('nonexistent.cubin')
            assert result is None

    def test_load_file_too_short(self):
        """Test loading a file that's too short to contain a hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            # Write a file with less than _hash_length bytes
            name = 'short.cubin'
            data = b'too_short'
            assert len(data) < _hash_length
            path = os.path.join(tmpdir, name)
            with open(path, 'wb') as f:
                f.write(data)

            result = backend.load(name)
            assert result is None

    def test_load_corrupted_hash(self):
        """Test that corrupted cache files are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            name = 'corrupted.cubin'
            path = os.path.join(tmpdir, name)

            # Write file with wrong hash
            cubin = b'kernel_data'
            wrong_hash = b'0' * _hash_length  # Wrong hash
            with open(path, 'wb') as f:
                f.write(wrong_hash + cubin)

            # Load should return None due to hash mismatch
            result = backend.load(name)
            assert result is None
