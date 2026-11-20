#!/usr/bin/env python3
"""
Check that a CuPy release has exactly the expected sdist and wheels.

The expected wheels are derived from the build matrix of the release
source tree (``--source-root``): ``ci/versions.yml`` (CUDA and Python
versions) and ``ci/tools/wheel_configs.py`` (package names and platforms).
"""
from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import distlib.locators
import yaml
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[2]

sdist_project = 'cupy'

# Host platform -> wheel platform. auditwheel picks the manylinux policy
# tags (possibly several), so those wheels match on the architecture alone.
_PLATFORMS = {
    'linux-64': 'manylinux*_x86_64',
    'linux-aarch64': 'manylinux*_aarch64',
    'win-64': 'win_amd64',
    'win-arm64': 'win_arm64',
}

# Not built in-tree, so neither required nor rejected (cupy/cupy#10491).
_UNCHECKED_PREFIXES = ('cupy_rocm_',)


def get_wheel_projects(source_root: Path) -> dict[str, list[tuple[str, str]]]:
    """Returns the (python-abi, platform) cells of each wheel project."""
    with open(source_root / 'ci' / 'versions.yml', encoding='UTF-8') as f:
        versions = yaml.safe_load(f)
    configs = runpy.run_path(
        str(source_root / 'ci' / 'tools' / 'wheel_configs.py'))

    pythons = []
    for python in versions['python']['supported']:
        tag = 'cp' + python.removesuffix('t').replace('.', '')
        pythons.append(f'{tag}-{tag}t' if python.endswith('t')
                       else f'{tag}-{tag}')

    projects = {}
    for kind in ('build', 'prev_build'):
        major = versions['cuda'][kind]['version'].split('.')[0]
        projects[configs['WHEEL_PACKAGE_NAMES'][major]] = [
            (python, _PLATFORMS[platform])
            for python in pythons
            for platform in configs['PRELOAD_LIBRARIES'][major]
        ]
    return projects


def get_basenames(project: str, version: str) -> list[str]:
    # List all wheels including unsupported ones by the current Python
    distlib.locators.is_compatible = lambda *_: True
    locator = distlib.locators.SimpleScrapingLocator(
        'https://pypi.org/simple/')
    proj: dict[str, str] = locator.get_project(project)['urls']
    if version not in proj:
        return []
    return [os.path.basename(url) for url in proj[version]]


def get_basenames_github(tag: str) -> list[str]:
    return subprocess.check_output([
        'gh', 'release', '--repo', 'cupy/cupy',
        'view', tag,
        '--json', 'assets',
        '--jq', '.assets[].name'], encoding='UTF-8').splitlines()


def get_basenames_local(path: str) -> list[str]:
    return [p.name for p in Path(path).iterdir() if p.is_file()]


def get_expected_sdist_key(project: str, version: str) -> str:
    return f'{project}-{version}.tar.gz'


def get_expected_wheel_key(
    project: str, version: str, abi: str, arch: str
) -> str:
    project = project.replace('-', '_')
    return f'{project}-{version}-{abi}-{arch}.whl'


def _normalize_platform(platform: str) -> str:
    for arch in ('x86_64', 'aarch64'):
        if platform.startswith('manylinux') and platform.endswith(f'_{arch}'):
            return f'manylinux*_{arch}'
    return platform


def get_asset_key(basename: str) -> str:
    """Maps a wheel to its expected key; other files map to themselves."""
    try:
        name, version, build, tags = parse_wheel_filename(basename)
    except InvalidWheelFilename:
        return basename
    keys = {
        get_expected_wheel_key(
            name, str(version), f'{tag.interpreter}-{tag.abi}',
            _normalize_platform(tag.platform))
        for tag in tags
    }
    if build or len(keys) != 1:
        return basename
    return keys.pop()


def verify(
    title: str, expected: Iterable[str], actual: Iterable[str]
) -> bool:
    print(f'🔵 Verifying release assets on {title}')
    expected = set(expected)
    found: dict[str, list[str]] = {}
    unchecked = []
    for basename in actual:
        if basename.startswith(_UNCHECKED_PREFIXES):
            unchecked.append(basename)
        else:
            found.setdefault(get_asset_key(basename), []).append(basename)
    error = False
    for p in sorted(expected - set(found)):
        error = True
        print(f'  ❓ Missing: {p}')
    for key in sorted(set(found) - expected):
        for p in sorted(found[key]):
            error = True
            print(f'  ⚠️  Unexpected: {p}')
    for key in sorted(set(found) & expected):
        if len(found[key]) > 1:
            error = True
            print(f'  ⚠️  Duplicate: {", ".join(sorted(found[key]))}')
        else:
            print(f'  👀 Found: {found[key][0]}')
    for p in sorted(unchecked):
        print(f'  ⏭️  Not checked: {p}')
    if error:
        print('  ❌ Check Fail')
    else:
        print('  ✅ Check Pass')
    print()
    return not error


def get_expected_wheels(
    wheel_projects: dict[str, list[tuple[str, str]]], version: str
) -> dict[str, list[str]]:
    return {
        project: [
            get_expected_wheel_key(project, version, cpython, arch)
            for (cpython, arch) in matrix
        ]
        for (project, matrix) in wheel_projects.items()
    }


class _LibAssetsArgs(argparse.Namespace):
    version: str
    source_root: Path
    local: str | None
    github: bool
    pypi_sdist: bool
    pypi_wheel: bool


def parse_args(argv: Sequence[str]) -> _LibAssetsArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument(
        '--source-root', type=Path, default=DEFAULT_SOURCE_ROOT,
        help='CuPy source tree of the release (default: this checkout)')
    parser.add_argument(
        '--local', metavar='DIR',
        help='check the distributions in a local directory')
    parser.add_argument('--github', action='store_true', default=False)
    parser.add_argument('--pypi-sdist', action='store_true', default=False)
    parser.add_argument('--pypi-wheel', action='store_true', default=False)
    return parser.parse_args(argv[1:], namespace=_LibAssetsArgs())


def main(argv: Sequence[str]) -> int:
    options = parse_args(argv)

    tag = 'v' + options.version.removeprefix('v')
    # Distribution file names carry the normalized version.
    version = str(Version(tag))
    wheel_projects = get_wheel_projects(options.source_root)
    expected_whl = get_expected_wheels(wheel_projects, version)
    expected_sdist = [get_expected_sdist_key(sdist_project, version)]
    expected_all = [
        *(key for keys in expected_whl.values() for key in keys),
        *expected_sdist,
    ]

    success = True

    # Verify distributions downloaded from a CI run
    if options.local is not None:
        success = verify(
            options.local,
            expected_all,
            get_basenames_local(options.local)) and success

    # Verify assets on GitHub release
    if options.github:
        success = verify(
            'GitHub Release',
            expected_all,
            get_basenames_github(tag)) and success

    if options.pypi_sdist:
        actual = get_basenames(sdist_project, version)
        success = verify(sdist_project, expected_sdist, actual) and success

    if options.pypi_wheel:
        for project in wheel_projects:
            actual = get_basenames(project, version)
            success = verify(
                f'PyPI: {project}', expected_whl[project], actual) and success

    if not success:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
