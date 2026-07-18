from __future__ import annotations

import datetime
import json
import subprocess
from typing import Any


CACHE_PREFIXES = (
    'static-checks-',
    'build-cuda-',
    'build-rocm-',
)
PR_CACHE_RETENTION = datetime.timedelta(days=1)


def _parse_datetime(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))


def _get_cache_prefix(key: str) -> str | None:
    for prefix in CACHE_PREFIXES:
        if key.startswith(prefix):
            return prefix
    return None


def _get_last_accessed_at(cache: dict[str, Any]) -> datetime.datetime:
    value = cache.get('last_accessed_at') or cache['created_at']
    return _parse_datetime(value)


def select_caches_to_delete(
        caches: list[dict[str, Any]],
        now: datetime.datetime,
) -> list[int]:
    cutoff = now - PR_CACHE_RETENTION
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for cache in caches:
        prefix = _get_cache_prefix(cache['key'])
        if prefix is None:
            continue
        groups.setdefault((prefix, cache['ref']), []).append(cache)

    caches_to_delete = []
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda cache: (
                _parse_datetime(cache['created_at']),
                cache['id'],
            ),
            reverse=True,
        )
        newest = ordered[0]
        if (newest['ref'].startswith('refs/pull/')
                and _get_last_accessed_at(newest) < cutoff):
            caches_to_delete.append(newest['id'])
        caches_to_delete.extend(cache['id'] for cache in ordered[1:])

    return sorted(caches_to_delete)


def _list_caches() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            'gh',
            'api',
            '--paginate',
            '--slurp',
            'repos/{owner}/{repo}/actions/caches?per_page=100',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pages = json.loads(completed.stdout)
    return [
        cache
        for page in pages
        for cache in page.get('actions_caches', [])
    ]


def _delete_cache(cache_id: int) -> None:
    completed = subprocess.run(
        ['gh', 'cache', 'delete', str(cache_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end='')
    if completed.returncode != 0 and completed.stderr:
        print(completed.stderr, end='')


def main() -> None:
    caches = _list_caches()
    caches_to_delete = select_caches_to_delete(
        caches, datetime.datetime.now(datetime.timezone.utc))

    if not caches_to_delete:
        print('No caches to delete')
        return

    print(f'Deleting {len(caches_to_delete)} cache(s)')
    for cache_id in caches_to_delete:
        _delete_cache(cache_id)


if __name__ == '__main__':
    main()
