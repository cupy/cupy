"""Avoid invalidating kernels when unused bundled headers change.

- Build: hash groups with transitive dependencies.
- Runtime: scan single-line literal includes, even in comments/strings;
  unknown includes/options select the full key.
- Trust portable relative <...> names and known options, checked before CuPy
  adds flags/paths. Assume toolchain headers/vendor macros don't pull in
  optional CuPy headers.
- Validate CuPy-owned dependencies at build time; reject fallbacks in tests.
- Limits: no macro expansion, concatenation, or line splicing. Nonliteral
  operands and embedded directive comments use the full key. Line-spliced
  directives can be missed; comments/strings can add false dependencies.
"""

from __future__ import annotations

import glob
import hashlib
import json
import posixpath
import re
from pathlib import Path


_GROUPS = {
    'xsf': re.compile(r'cupy/xsf/'),
    'float16': re.compile(r'cupy/float16\.cuh$'),
    'bfloat16': re.compile(r'cupy/bfloat16\.cuh$'),
    'structview': re.compile(r'cupy/structview\.cuh$'),
    'unstructured_void': re.compile(r'cupy/unstructured_void\.cuh$'),
    'jitify': re.compile(r'cupy/_jitify/'),
    # Build-only in CuPy, but still cover explicit includes in user kernels.
    # TODO(seberg): Consider excluding these headers from wheels?
    'build': re.compile(r'cupy/(?:type_dispatcher\.cuh$|_dlpack/)'),
}

_RELATIVE_HEADER_RE = re.compile(r'[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*')

# Embedded comments after the marker take the full-key fallback.
# We intentionally ignore line splicing after the marker or inside directive
# keywords. These theoretical cases can hide includes from the scanner.
_INCLUDE_PATTERN = r'[ \t\v\f]*(?:(include_next|include|import)\b(.+)|/)'
_INCLUDE_RE = re.compile(r'(?:#|%:|\?\?=)' + _INCLUDE_PATTERN)
_INCLUDE_RE_PREFIX = re.compile('#' + _INCLUDE_PATTERN)
_LITERAL_RE = re.compile(r'\s*(?:<([^>]+)>|"([^"]+)")\s*(?://.*)?')
_CCCL_PREFIXES = ('cupy/_cccl/cub/', 'cupy/_cccl/thrust/',
                  'cupy/_cccl/libcudacxx/')

# Enabled by the test suite to detect CuPy kernels taking the fallback.
_error_on_unknown_header = False


class _UnknownHeaderError(RuntimeError):
    pass


def load_header_cache():
    return json.loads(Path(__file__).with_suffix('.json').read_text())


def _include_name(header):
    for prefix in _CCCL_PREFIXES:
        if header.startswith(prefix):
            return header[len(prefix):]
    return header


def _get_header_groups(source, headers, parent='', *, unknown='full'):
    """Collect groups; unresolved includes use 'full', 'error', or 'ignore'."""
    groups = set()
    include_re = (_INCLUDE_RE if '%:' in source or '??=' in source
                  else _INCLUDE_RE_PREFIX)
    for match in include_re.finditer(source):
        literal = (_LITERAL_RE.fullmatch(match[2])
                   if match[1] == 'include' else None)
        if literal is not None:
            header = literal[1] or literal[2]
            quoted = literal[2] is not None
            if quoted and parent:
                relative = posixpath.normpath(posixpath.join(parent, header))
                relative = _include_name(relative)
                if relative in headers:
                    groups.add(headers[relative])
                    continue
            header = _include_name(header)
            if header in headers:
                groups.add(headers[header])
                continue
            if (not quoted and not header.startswith('cupy/')
                    and _RELATIVE_HEADER_RE.fullmatch(header)
                    and not {'.', '..'}.intersection(header.split('/'))):
                continue
        if unknown == 'ignore':
            continue
        if unknown == 'error' or _error_on_unknown_header:
            raise _UnknownHeaderError(
                f'Cannot classify kernel include: {match[0]}')
        return {'full'}
    return groups


def get_cache_key(source, cache, options=(), jitify=False):
    """Select group keys before internal compiler flags/paths are added."""
    for option in options:
        if not option.startswith((
                '-D', '--define-macro=', '-std', '--std', '-O', '-w')):
            if _error_on_unknown_header:
                raise _UnknownHeaderError(
                    f'Cannot classify compiler option: {option}')
            return cache['keys']['full']
    groups = _get_header_groups(source, cache['headers'])
    if 'full' in groups:
        return cache['keys']['full']
    groups.add('common')
    if jitify:
        groups.add('jitify')
    return ' '.join([f'{group}:{cache["keys"][group]}'
                     for group in sorted(groups)])


def build_header_cache(include_dir):
    """Hash groups with transitive dependencies, including cycles."""
    include_dir = Path(include_dir)
    files = {}
    headers = {}
    for filename in sorted(glob.glob(str(include_dir / '**'), recursive=True)):
        path = Path(filename)
        if (not path.is_file() or path.suffix in {'.md', '.rst'}
                or path.name.startswith(('LICENSE', 'README', '.'))):
            continue
        name = path.relative_to(include_dir).as_posix()
        group = next((g for g, pattern in _GROUPS.items()
                      if pattern.match(name)), 'common')
        files[name] = (path, group)
        # Match the search paths supplied by add_cupy_include_dirs.
        headers[_include_name(name)] = group

    hashes = {g: hashlib.sha1(usedforsecurity=False)
              for g in ('common', *_GROUPS)}
    dependencies = {g: set() for g in hashes}
    # Changes to classification policy must invalidate existing entries too.
    hashes['common'].update(Path(__file__).read_bytes())
    for name, (path, group) in files.items():
        content = path.read_bytes()
        hashes[group].update(name.encode() + b'\0' + content + b'\0')
        vendor = name.startswith((
            'cupy/_cccl/', 'cupy/_jitify/', 'cupy/_dlpack/', 'cupy/xsf/'))
        try:
            dependencies[group].update(_get_header_groups(
                content.decode(), headers, posixpath.dirname(name),
                unknown='ignore' if vendor else 'error'))
        except _UnknownHeaderError as error:
            raise _UnknownHeaderError(f'{name}: {error}') from None

    own_keys = {g: h.hexdigest() for g, h in hashes.items()}

    def combine(groups):
        content = '\0'.join(f'{g}:{own_keys[g]}' for g in sorted(groups))
        return hashlib.sha1(
            content.encode(), usedforsecurity=False).hexdigest()

    keys = {}
    for group in hashes:
        reachable = {group}
        pending = [group]
        while pending:
            for dependency in dependencies[pending.pop()] - reachable:
                reachable.add(dependency)
                pending.append(dependency)
        keys[group] = combine(reachable)
    keys['full'] = combine(hashes)
    return {'keys': keys, 'headers': headers}
