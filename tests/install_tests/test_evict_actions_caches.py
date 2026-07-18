from __future__ import annotations

import datetime
import importlib.util
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / '.github/workflows/scripts/evict_actions_caches.py'
    )
    spec = importlib.util.spec_from_file_location(
        'evict_actions_caches', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evict_actions_caches = _load_module()


def _cache(
        cache_id: int,
        key: str,
        ref: str,
        created_at: str,
        last_accessed_at: str | None = None,
) -> dict[str, object]:
    if last_accessed_at is None:
        last_accessed_at = created_at
    return {
        'id': cache_id,
        'key': key,
        'ref': ref,
        'created_at': created_at,
        'last_accessed_at': last_accessed_at,
    }


def test_select_caches_to_delete_keeps_latest_per_prefix_and_branch() -> None:
    now = datetime.datetime(2026, 7, 18, tzinfo=datetime.timezone.utc)
    caches = [
        _cache(
            1,
            'static-checks-3.10-old',
            'refs/heads/main',
            '2026-07-16T00:00:00Z',
        ),
        _cache(
            2,
            'static-checks-3.10-new',
            'refs/heads/main',
            '2026-07-17T00:00:00Z',
        ),
        _cache(
            3,
            'build-cuda-old',
            'refs/heads/feature',
            '2026-07-15T00:00:00Z',
        ),
        _cache(
            4,
            'build-cuda-new',
            'refs/heads/feature',
            '2026-07-18T00:00:00Z',
        ),
        _cache(
            5,
            'unrelated-cache',
            'refs/heads/main',
            '2026-07-17T00:00:00Z',
        ),
    ]

    assert evict_actions_caches.select_caches_to_delete(caches, now) == [1, 3]


def test_select_caches_to_delete_expires_old_pull_request_caches() -> None:
    now = datetime.datetime(2026, 7, 18, tzinfo=datetime.timezone.utc)
    caches = [
        _cache(
            1,
            'build-rocm-old',
            'refs/pull/1/merge',
            '2026-07-16T00:00:00Z',
            '2026-07-16T12:00:00Z',
        ),
        _cache(
            2,
            'build-rocm-newest-but-stale',
            'refs/pull/1/merge',
            '2026-07-17T00:00:00Z',
            '2026-07-16T18:00:00Z',
        ),
        _cache(
            3,
            'build-cuda-old',
            'refs/pull/2/merge',
            '2026-07-17T02:00:00Z',
            '2026-07-17T06:00:00Z',
        ),
        _cache(
            4,
            'build-cuda-newest',
            'refs/pull/2/merge',
            '2026-07-18T00:00:00Z',
            '2026-07-18T01:00:00Z',
        ),
    ]

    assert evict_actions_caches.select_caches_to_delete(caches, now) == [
        1, 2, 3,
    ]
