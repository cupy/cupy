#!/usr/bin/env python

"""
A tool to clean-up least-recently-used CuPy kernel caches.

The expiry is the length to invalidate unused CuPy cache, in seconds.
For example, setting the expiry to 3,600 will trim all kernel caches
not being used within an hour. The maximum number of files or maximum total
size can also be specified to cap the amount of caches.

Note that this code relies on atime support of the filesystem.
"""

import argparse
import datetime
import glob
import itertools
import os
import sys


def _log(msg):
    sys.stderr.write(msg + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expiry', type=int, default=None)
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--max-size', type=int, default=None)
    parser.add_argument('--print', default=False, action='store_true')
    parser.add_argument('--rm', default=False, action='store_true')
    options = parser.parse_args()

    cache_dir = os.environ.get(
        'CUPY_CACHE_DIR',
        os.path.expanduser('~/.cupy/kernel_cache/'))
    _log('Looking for cache files under {}...'.format(cache_dir))
    records = []
    for f in itertools.chain(
            glob.iglob(os.path.join(cache_dir, '*.cubin')),
            glob.iglob(os.path.join(cache_dir, '*.hsaco'))):
        stat = os.lstat(f)
        records.append((stat.st_atime, stat.st_size, f))
    records.sort(reverse=True)  # newest cache first

    if options.expiry:
        expiry = datetime.datetime.now() - datetime.timedelta(
            seconds=options.expiry)
        expiry_ts = expiry.timestamp()
    else:
        expiry_ts = None

    keep_count = 0
    keep_size = 0
    for (atime, size, _) in records:
        if ((expiry_ts is not None and atime < expiry_ts) or
            (options.max_files is not None and
                options.max_files <= keep_count) or
            (options.max_size is not None and
                options.max_size < (keep_size + size))):
            break
        keep_count += 1
        keep_size += size

    total_count = len(records)
    trim_size = 0
    for i in range(keep_count, total_count):
        _, size, f = records[i]
        trim_size += size
        if options.print:
            print(f)
        if options.rm:
            os.unlink(f)

    _log('Total:   {} bytes, {} files'.format(
        keep_size + trim_size, total_count))
    _log('Valid:   {} bytes, {} files'.format(
        keep_size, keep_count))
    _log('Expired: {} bytes, {} files'.format(
        trim_size, total_count - keep_count))


if __name__ == '__main__':
    main()
