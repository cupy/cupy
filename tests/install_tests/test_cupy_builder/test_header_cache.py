"""Header-cache policy tests that do not require built CuPy extensions."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest


pytestmark = pytest.mark.thread_unsafe(reason='Changes header-cache policy')


_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def header_cache_keys(tmp_path):
    # The builder locates include/ beside this file; avoid importing CuPy.
    path = tmp_path / '_header_cache_keys.py'
    shutil.copyfile(_ROOT / 'cupy/_core/_header_cache_keys.py', path)
    spec = importlib.util.spec_from_file_location(
        '_cupy_header_cache_keys', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def get_cache_key(header_cache_keys, monkeypatch):
    """Select keys using each test's generated metadata."""
    def get(source, cache, options=(), *, jitify=False):
        monkeypatch.setattr(header_cache_keys, 'HEADER_CACHE_KEYS', cache,
                            raising=False)
        return header_cache_keys.get_cache_key(source, options, jitify=jitify)
    return get


def _write(root, name, source):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


@pytest.fixture
def include_dir(tmp_path):
    tmp_path = tmp_path / 'include'
    _write(tmp_path, 'cupy/carray.cuh', '// common\n')
    for dtype in ('float16', 'bfloat16', 'structview', 'unstructured_void'):
        _write(tmp_path, f'cupy/{dtype}.cuh', '#include "cupy/carray.cuh"\n')
    _write(tmp_path, 'cupy/xsf/function.h', '// xsf\n')
    return tmp_path


@pytest.fixture
def cache(include_dir, header_cache_keys):
    return header_cache_keys.build_header_cache_keys(hip=False)


@pytest.mark.parametrize('group', [
    'xsf', 'float16', 'bfloat16', 'structview', 'unstructured_void',
])
def test_optional_update(include_dir, group, header_cache_keys, get_cache_key):
    before = header_cache_keys.build_header_cache_keys(hip=False)
    name = 'cupy/xsf/function.h' if group == 'xsf' else f'cupy/{group}.cuh'
    with (include_dir / name).open('a') as f:
        f.write('// changed\n')
    after = header_cache_keys.build_header_cache_keys(hip=False)
    assert before['keys']['common'] == after['keys']['common']
    assert before['keys'][group] != after['keys'][group]
    assert before['keys']['full'] != after['keys']['full']
    source = f'#include <{name}>'
    assert (get_cache_key(source, before)
            != get_cache_key(source, after))
    assert (get_cache_key('', before)
            == get_cache_key('', after))
    # External wrappers must invalidate even though their contents are unknown.
    source = '#include "user_wrapper.h"'
    assert (get_cache_key(source, before)
            != get_cache_key(source, after))


def test_transitive_dependencies_and_cycles(include_dir, header_cache_keys):
    _write(include_dir, 'cupy/wrapper.h', '#include "detail/other.h"\n')
    _write(include_dir, 'cupy/detail/other.h',
           '#if SOME_MACRO\n#include "../float16.cuh"\n#endif\n')
    _write(include_dir, 'cupy/float16.cuh', '#include "bfloat16.cuh"\n')
    _write(include_dir, 'cupy/bfloat16.cuh',
           '#include "float16.cuh"\n#include "xsf/function.h"\n')
    before = header_cache_keys.build_header_cache_keys(hip=False)
    _write(include_dir, 'cupy/xsf/function.h', '// changed\n')
    after = header_cache_keys.build_header_cache_keys(hip=False)
    for group in ('common', 'float16', 'bfloat16', 'xsf'):
        assert before['keys'][group] != after['keys'][group]


def test_build_only_and_metadata(include_dir, header_cache_keys):
    _write(include_dir, 'cupy/_dlpack/dlpack.h', '// dlpack\n')
    _write(include_dir, 'cupy/type_dispatcher.cuh', '// dispatcher\n')
    before = header_cache_keys.build_header_cache_keys(hip=False)
    _write(include_dir, 'cupy/README.md', 'Documentation\n')
    _write(include_dir, 'cupy/_dlpack/LICENSE', 'License\n')
    assert before == header_cache_keys.build_header_cache_keys(hip=False)
    _write(include_dir, 'cupy/_dlpack/dlpack.h', '// changed\n')
    after = header_cache_keys.build_header_cache_keys(hip=False)
    assert before['keys']['common'] == after['keys']['common']
    assert before['keys']['build'] != after['keys']['build']
    assert before['keys']['full'] != after['keys']['full']


@pytest.mark.parametrize('directive', [
    '#include <cupy/float16.cuh>',
    '  # include "cupy/float16.cuh" // comment',
    '/* comment */ #include <cupy/float16.cuh>',
    '%:include <cupy/float16.cuh>',
    '??=include <cupy/float16.cuh>',
])
def test_literal_includes(cache, directive, get_cache_key):
    key = get_cache_key(directive, cache)
    assert f'float16:{cache["keys"]["float16"]}' in key
    assert 'bfloat16:' not in key


def test_comments(cache, get_cache_key):
    source = ('/*\n#include <cupy/bfloat16.cuh>\n*/\n'
              '#include <cupy/float16.cuh>\n')
    key = get_cache_key(source, cache)
    assert f'float16:{cache["keys"]["float16"]}' in key
    assert f'bfloat16:{cache["keys"]["bfloat16"]}' in key


@pytest.mark.parametrize('source', [
    '#include "user_wrapper.h"',
    '#define HEADER "user.h"\n#include HEADER',
    '#include_next <stdint.h>',
    '#include <cupy/unknown_header.h>',
    '#include "stdint.h"',
    '#include </tmp/wrapper.h>',
    '#include <C:/headers/wrapper.h>',
    r'#include <C:\headers\wrapper.h>',
    r'#include <\\server\share\wrapper.h>',
    '#include <C:wrapper.h>',
    '#include <../wrapper.h>',
    '#include <dir/../wrapper.h>',
    '#include <dir//wrapper.h>',
    '#include <a b.h>',
    '#/**/include <cupy/float16.cuh>',
    '#include /* comment */ <cupy/float16.cuh>',
    '#include "cupy/float16.cuh" "cupy/bfloat16.cuh"',
    '#include <cupy/float16.\ncuh>',
    '#include \\\n<cupy/float16.cuh>',
    '#include \\\r\n<cupy/float16.cuh>',
])
def test_unknown_include(cache, source, get_cache_key):
    assert get_cache_key(source, cache) == cache['keys']['full']


def test_strict_mode(cache, monkeypatch, header_cache_keys, get_cache_key):
    monkeypatch.setattr(header_cache_keys, '_error_on_unknown_header', True)
    with pytest.raises(header_cache_keys._UnknownHeaderError,
                       match='Cannot classify kernel include'):
        get_cache_key('#include "user.h"', cache)


def test_unknown_options(cache, monkeypatch, header_cache_keys, get_cache_key):
    options = ('--pre-include=user.h',)
    assert (get_cache_key('', cache, options)
            == cache['keys']['full'])
    monkeypatch.setattr(header_cache_keys, '_error_on_unknown_header', True)
    with pytest.raises(header_cache_keys._UnknownHeaderError,
                       match='compiler option'):
        get_cache_key('', cache, options)


def test_known_options(cache, get_cache_key):
    source = '#include <stdint.h>\n#include <cupy/float16.cuh>'
    options = ('-std=c++20', '--std=c++23', '-O3', '-Ofast', '-w',
               '-DUSER_MACRO=1', '--define-macro=OTHER=2')
    assert (get_cache_key(source, cache, options)
            == get_cache_key(source, cache))


def test_toolchain_header_names(cache, get_cache_key):
    source = ('#include <new_system_header.h>\n'
              '#include <cuda/unknown_header>\n'
              '#include <dir/header++.h>\n')
    assert (get_cache_key(source, cache)
            == get_cache_key('', cache))


def test_jitify_update(include_dir, header_cache_keys, get_cache_key):
    _write(include_dir, 'cupy/_jitify/jitify.hpp', '// jitify\n')
    before = header_cache_keys.build_header_cache_keys(hip=False)
    _write(include_dir, 'cupy/_jitify/jitify.hpp', '// changed\n')
    after = header_cache_keys.build_header_cache_keys(hip=False)
    assert (get_cache_key('', before)
            == get_cache_key('', after))
    assert (get_cache_key('', before, jitify=True)
            != get_cache_key('', after, jitify=True))


def test_unknown_owned_header(include_dir, header_cache_keys):
    _write(include_dir, 'cupy/wrapper.h', '#include "unknown.h"')
    with pytest.raises(header_cache_keys._UnknownHeaderError,
                       match='cupy/wrapper.h:.*unknown.h'):
        header_cache_keys.build_header_cache_keys(hip=False)
