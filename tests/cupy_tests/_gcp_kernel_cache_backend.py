"""
GCP Cloud Storage backed kernel cache for CuPy.

This module provides an **EXPERIMENTAL** cache backend that stores compiled
kernel binaries in Google Cloud Platform (GCP) Cloud Storage.

**WARNING**: This is an experimental feature and currently intended solely for
use with CuPy's CI environment.

Requirements:
    - google-cloud-storage: Install with `pip install google-cloud-storage`
    - GCP credentials configured (via GOOGLE_APPLICATION_CREDENTIALS or
      gcloud auth application-default login)
"""

from __future__ import annotations
from cupy_backends.cuda.libs import nvrtc as _nvrtc
from cupy.cuda.compiler import _get_cupy_cache_key

import platform
import sys
import time
import warnings

_GCP_AVAILABLE = True
try:
    from google.cloud import storage
    from google.cloud.storage import transfer_manager
    from google.api_core import exceptions as gcp_exceptions
except ImportError:
    _GCP_AVAILABLE = False
    storage = None  # type: ignore
    transfer_manager = None  # type: ignore
    gcp_exceptions = None  # type: ignore

from cupy.cuda._compiler_cache import DiskKernelCacheBackend  # noqa


def _get_platform_subdir() -> str:
    """Return the conda-style platform subdir string (e.g. ``linux-64``)."""
    machine = platform.machine()
    if sys.platform.startswith('linux'):
        if machine == 'x86_64':
            return 'linux-64'
        if machine in ('aarch64', 'arm64'):
            return 'linux-aarch64'
        assert False
    if sys.platform.startswith('win'):
        return 'win-64'
    assert False


class GCPStorageCacheBackend(DiskKernelCacheBackend):
    """
    GCP Cloud Storage backed cache backend.

    This cache backend stores compiled kernel binaries in Google Cloud Storage
    for sharing across distributed environments. Downloaded files are persisted
    to local disk for faster future access.

    Args:
        local_cache_dir (str | None, optional): Local directory to cache
            downloaded files. Defaults to ~/.cupy/kernel_cache.
        bucket_name (str): Name of the GCS bucket to use for cache storage.
        prefix (str): Prefix to use for all cache keys in GCS.
        project (str | None, optional): GCP project ID. If None, uses the
            default project from credentials.

    Attributes:
        bucket_name (str): The GCS bucket name.
        prefix (str): The GCS key prefix for cache entries.
    """

    def __init__(
        self,
        *,
        local_cache_dir: str | None = None,
        bucket_name: str,
        prefix: str,
        project: str | None = None,
    ) -> None:
        """Initialize the GCP Storage cache backend."""
        # Initialize the parent disk cache
        super().__init__(local_cache_dir)

        self._gcp_enabled = False
        self._prefix = prefix
        # Narrow the GCS namespace to this pipeline's cache key and NVRTC
        # (CUDA) version so that initialize_local_cache only downloads what
        # this pipeline needs, and so that kernels compiled with different
        # CuPy builds do not collide or interfere with each other.
        major, minor = _nvrtc.getVersion()
        self._nvrtc_prefix = (
            f'{self._prefix}{_get_cupy_cache_key()}/'
            f'{_get_platform_subdir()}/CUDA_{major}_{minor}/'
        )
        if not _GCP_AVAILABLE:
            warnings.warn(
                "google-cloud-storage is not installed. "
                "GCPStorageCacheBackend will only use local disk cache. "
                "Install with: pip install google-cloud-storage",
                RuntimeWarning
            )
            return

        try:
            self._client = storage.Client(project=project)
            self._bucket = self._client.bucket(bucket_name)
        except Exception as e:
            warnings.warn(
                f"Failed to initialize GCS client: {type(e)}: {e}; "
                "Will only use local disk cache.",
                RuntimeWarning
            )
            return

        self._gcp_enabled = True

    def initialize_local_cache(self, *, max_workers: int = 16) -> int:
        """
        Pre-populate local disk cache by bulk-downloading all kernel files
        from GCS.

        Call this once at startup to warm the local cache, so that subsequent
        kernel lookups during the test run can be served from disk rather than
        triggering individual GCS network requests.

        Args:
            max_workers (int): Number of parallel download threads.
                Defaults to 16.

        Returns:
            int: Number of new files downloaded from GCS.
        """
        if not self._gcp_enabled:
            return 0

        t0 = time.perf_counter()
        prefix_len = len(self._nvrtc_prefix)

        try:
            # list_blobs returns a lazy HTTPIterator; materialise it to
            # separate listing errors from download errors and to build the
            # list passed to download_many_to_path.
            blob_iter = self._bucket.list_blobs(prefix=self._nvrtc_prefix)
            # blob_names are relative to self._nvrtc_prefix (passed separately
            # as blob_name_prefix so it is stripped from the destination path).
            blob_names = [blob.name[prefix_len:] for blob in blob_iter]
        except Exception as e:
            warnings.warn(
                f"Failed to list GCS objects for cache initialization: "
                f"{type(e)}: {e}",
                RuntimeWarning
            )
            return 0

        if not blob_names:
            return 0

        try:
            # GCS stores hash+cubin (same format as disk), so
            # download_many_to_path writes files that
            # DiskKernelCacheBackend.load() can read directly.
            transfer_manager.download_many_to_path(
                self._bucket,
                blob_names,
                destination_directory=self._cache_dir,
                blob_name_prefix=self._nvrtc_prefix,
                worker_type=transfer_manager.THREAD,
                max_workers=max_workers,
                raise_exception=True,
            )
        except Exception as e:
            warnings.warn(
                f"Failed to download GCS objects for cache initialization: "
                f"{type(e)}: {e}",
                RuntimeWarning
            )
            return 0

        downloaded = len(blob_names)
        elapsed = time.perf_counter() - t0
        print(
            f"GCP kernel cache: {downloaded} new file(s) downloaded "
            f"in {elapsed:.1f}s.",
            flush=True,
        )
        return downloaded

    def load(self, name: str) -> bytes | None:
        """
        Load a cached kernel binary.

        First checks local disk cache, then falls back to GCS if not found.
        Downloaded files are persisted to local disk for future use.

        Args:
            name (str): The cache key (filename) for the compiled kernel.

        Returns:
            bytes or None: The cubin binary data (without hash prefix) if
                found and valid, None otherwise.
        """
        # First, try to load from local disk cache
        data = super().load(name)
        if data is not None:
            return data

        if not self._gcp_enabled:
            return None
        try:
            blob = self._bucket.blob(self._nvrtc_prefix + name)
            data = blob.download_as_bytes()
        except gcp_exceptions.NotFound:
            # Cache miss.
            return None
        except Exception as e:
            warnings.warn(
                f"Failed to download from GCS: {type(e)}: {e}",
                RuntimeWarning
            )
            return None

        # GCS stores hash+cubin (same as disk format); validate and strip hash.
        cubin = self._decode_cubin(data)
        if cubin is None:
            return None

        # Persist to disk in the native on-disk format so that the next
        # DiskKernelCacheBackend.load() call reads it directly.
        self._write_encoded(name, data)

        return cubin

    def save(self, name: str, cubin: bytes, source: str) -> None:
        """
        Save a compiled kernel binary to cache.

        Saves to both local disk and GCS (if enabled).

        Args:
            name (str): The cache key (filename) for the compiled kernel.
            cubin (bytes): The compiled kernel binary data.
            source (str): The CUDA source code.
        """
        # Always save to local disk first
        super().save(name, cubin, source)

        if not self._gcp_enabled:
            return
        try:
            blob = self._bucket.blob(self._nvrtc_prefix + name)
            # Store hash+cubin in GCS (same as disk format) so that
            # transfer_manager.download_many_to_path() writes files that
            # DiskKernelCacheBackend.load() can read directly.
            blob.upload_from_string(self._encode_cubin(cubin))
        except Exception as e:
            warnings.warn(f"Failed to save to GCS: {e}.", RuntimeWarning)
            return None
