#!/usr/bin/env python3
"""Resolve mini-CTK components from NVIDIA redistrib metadata."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HOST_PLATFORM_TO_SUBDIR: dict[str, str] = {
    "linux-64": "linux-x86_64",
    "linux-aarch64": "linux-sbsa",
    "win-64": "windows-x86_64",
}

# CTK 13.3.0 renamed the redistrib key from cuda_cccl to cccl.
COMPONENT_ALIASES: dict[str, tuple[str, ...]] = {
    "cuda_cccl": ("cccl",),
}

def host_platform_to_subdir(host_platform: str) -> str:
    try:
        return HOST_PLATFORM_TO_SUBDIR[host_platform]
    except KeyError as exc:
        raise ValueError(f"unsupported host-platform: {host_platform!r}") from exc

def split_components(components: str) -> list[str]:
    return [component for component in components.split(",") if component]

def filter_static_components(components: list[str], host_platform: str, cuda_version: str) -> list[str]:
    try:
        cuda_major = int(cuda_version.split(".", 1)[0])
    except ValueError as exc:
        raise ValueError(f"invalid cuda-version: {cuda_version!r}") from exc

    filtered = []
    for component in components:
        if component == "libnvjitlink" and cuda_major < 12:
            continue
        if component in {"cuda_crt", "libnvvm"} and cuda_major < 13:
            continue
        if component == "libcufile" and host_platform.startswith("win-"):
            continue
        filtered.append(component)
    return filtered

def validate_metadata_url(metadata_url: str) -> str:
    parsed = urllib.parse.urlsplit(metadata_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"metadata URL must be an https URL: {metadata_url!r}")
    return metadata_url

def load_metadata(*, metadata_path: str | None, metadata_url: str | None) -> dict[str, Any]:
    if (metadata_path is None) == (metadata_url is None):
        raise ValueError("exactly one of --metadata-path or --metadata-url is required")

    if metadata_path is not None:
        return json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    assert metadata_url is not None
    metadata_url = validate_metadata_url(metadata_url)
    with urllib.request.urlopen(metadata_url) as response:  # noqa: S310 - scheme is restricted to https above
        return json.load(response)

def resolve_component_name(metadata: dict[str, Any], component: str) -> str:
    if component in metadata:
        return component

    for alias in COMPONENT_ALIASES.get(component, ()):
        if alias in metadata:
            return alias

    return component

def filter_components(
    metadata: dict[str, Any],
    *,
    host_platform: str,
    cuda_version: str,
    components: str,
) -> tuple[list[str], list[str]]:
    ctk_subdir = host_platform_to_subdir(host_platform)
    filtered = []
    skipped = []
    for component in filter_static_components(split_components(components), host_platform, cuda_version):
        resolved_component = resolve_component_name(metadata, component)
        if ctk_subdir in metadata.get(resolved_component, {}):
            filtered.append(resolved_component)
        else:
            skipped.append(component)
    return filtered, skipped

def get_component_relative_path(metadata: dict[str, Any], *, host_platform: str, component: str) -> str:
    ctk_subdir = host_platform_to_subdir(host_platform)
    component = resolve_component_name(metadata, component)
    component_info = metadata.get(component)
    if component_info is None:
        raise KeyError(f"unknown CTK component {component!r}")

    subdir_info = component_info.get(ctk_subdir)
    if subdir_info is None:
        raise KeyError(f"CTK component {component!r} is not available for redistrib subdir {ctk_subdir!r}")

    relative_path = subdir_info.get("relative_path")
    if relative_path is None:
        raise KeyError(f"CTK component {component!r} for redistrib subdir {ctk_subdir!r} is missing 'relative_path'")
    return relative_path

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser("filter-components")
    filter_parser.add_argument("--host-platform", required=True)
    filter_parser.add_argument("--cuda-version", required=True)
    filter_parser.add_argument("--components", required=True)
    filter_parser.add_argument("--metadata-path")
    filter_parser.add_argument("--metadata-url")

    relpath_parser = subparsers.add_parser("component-relative-path")
    relpath_parser.add_argument("--host-platform", required=True)
    relpath_parser.add_argument("--component", required=True)
    relpath_parser.add_argument("--metadata-path")
    relpath_parser.add_argument("--metadata-url")

    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        metadata = load_metadata(metadata_path=args.metadata_path, metadata_url=args.metadata_url)

        if args.command == "filter-components":
            filtered, skipped = filter_components(
                metadata,
                host_platform=args.host_platform,
                cuda_version=args.cuda_version,
                components=args.components,
            )
            for component in skipped:
                print(
                    f"Skipping unsupported CTK component {component!r} for host-platform {args.host_platform!r}",
                    file=sys.stderr,
                )
            print(",".join(filtered))
            return 0

        if args.command == "component-relative-path":
            print(
                get_component_relative_path(
                    metadata,
                    host_platform=args.host_platform,
                    component=args.component,
                )
            )
            return 0

        raise AssertionError(f"unexpected command: {args.command!r}")
    except (ValueError, KeyError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
