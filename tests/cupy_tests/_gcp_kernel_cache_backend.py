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

import warnings

_GCP_AVAILABLE = True
try:
    from google.cloud import storage
    from google.api_core import exceptions as gcp_exceptions
except ImportError:
    _GCP_AVAILABLE = False
    storage = None  # type: ignore
    gcp_exceptions = None  # type: ignore

from cupy.cuda._compiler_cache import DiskKernelCacheBackend  # noqa


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
            blob = self._bucket.blob(self._prefix + name)
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

        # Persist to local disk for future use
        # We pass empty string for source as we don't have it
        super().save(name, data, "")

        return data

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
            blob = self._bucket.blob(self._prefix + name)
            blob.upload_from_string(cubin)
        except Exception as e:
            warnings.warn(f"Failed to save to GCS: {e}.", RuntimeWarning)
            return None
