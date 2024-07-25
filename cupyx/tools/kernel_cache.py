#!/usr/bin/env python

import argparse
import datetime
import glob
import itertools
import os
import sys


def _log(msg):
    if sys.stderr is not None:
        sys.stderr.write(msg + '\n')
        sys.stderr.flush()


def _main(argv):
    parser = argparse.ArgumentParser(
        prog=f'{os.path.basename(sys.executable)} -m cupyx.tools.kernel_cache',
        description='''
CuPy Kernel Cache Management Tool
=================================

A tool to cleanup CuPy kernel caches.

CuPy caches compiled kernels to a file system (`~/.cupy/kernel_cache` or
path set via ${CUPY_CACHE_DIR}.)  This tool allows you to selectively remove
them to maintain caches in an optimal size.

There are three criterias available to filter cache files to be removed:

  * Maximum number of cache files
  * Total size of cache files
  * Expiry time

For example, when the maximum number of files is specified to 100, caches are
purged except for 100 recently used ones.  Setting the expiry to 3600 will
remove all kernel caches not being used within an hour.  When multiple
criterias are specified, they are ANDed.

Note that this tool relies on the timestamp recorded on a file system;
the maximum value of creation, modification, and access time is considered as
the time the cache was last used.

This tool dry-runs by default. Pass `--rm` option to actually remove files.
    ''',
    epilog='''
Examples:

  Show cache statistics:
    $ %(prog)s

  Remove outdated cache files that exceeds 1 GiB:
    $ %(prog)s --max-size 1073741824 --rm
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--max-files', type=int, default=None,
        help='Maximum number of cache files')
    parser.add_argument(
        '--max-size', type=int, default=None,
        help='Maximum total size of cache files (in bytes)')
    parser.add_argument(
        '--expiry', type=int, default=None,
        help='Cache expiry (in seconds)')
    parser.add_argument(
        '--print', default=False, action='store_true',
        help='Print path of cache files selected for removal')
    parser.add_argument(
        '--rm', default=False, action='store_true',
        help='Remove selected cache files')
    parser.add_argument(
        '--quiet', default=False, action='store_true',
        help='Suppress progress and statistics output')
    options = parser.parse_args(argv)

    _run(options, datetime.datetime.now())


def _run(options, timestamp):
    cache_dir = os.environ.get(
            'CUPY_CACHE_DIR',
            os.path.expanduser('~/.cupy/kernel_cache/'))
    if not options.quiet:
        _log('Looking for cache files...: {}'.format(cache_dir))

    records = []
    for fpath in itertools.chain(
            glob.iglob(os.path.join(cache_dir, '*.cubin')),
            glob.iglob(os.path.join(cache_dir, '*.hsaco'))):
        stat = os.lstat(fpath)
        records.append((
            max(stat.st_atime, stat.st_mtime, stat.st_ctime),
            stat.st_size,
            fpath))
    records.sort(reverse=True)  # newest cache first

    if options.expiry:
        expiry_ts = (timestamp - datetime.timedelta(
            seconds=options.expiry)).timestamp()
    else:
        expiry_ts = None

    keep_count = 0
    keep_size = 0
    for (time, size, _) in records:
        if ((expiry_ts is not None and time < expiry_ts) or
            (options.max_files is not None and
                options.max_files <= keep_count) or
            (options.max_size is not None and
                options.max_size < (keep_size + size))):
            break
        keep_count += 1
        keep_size += size

    total_count = len(records)
    trim_size = 0
    for (_, size, fpath) in records[keep_count:]:
        trim_size += size
        if options.print:
            print(fpath)
        if options.rm:
            os.unlink(fpath)

    if not options.quiet:
        _log('Total:   {} bytes, {} files'.format(
            keep_size + trim_size, total_count))
        _log('Alive:   {} bytes, {} files'.format(
            keep_size, keep_count))
        _log('Expired: {} bytes, {} files'.format(
            trim_size, total_count - keep_count))
        if trim_size != 0 and not options.rm:
            _log('(hint: use --rm option to remove expired cache files)')


if __name__ == '__main__':
    _main(sys.argv[1:])
