#!/usr/bin/env python

"""
A tool to clean-up unused CuPy kernel caches.

The expiry is the length to invalidate unused CuPy cache, in seconds.
For example, setting the expiry to 3,600 will trim all kernel caches
not being used within an hour.

Note that this code relies on atime support of the filesystem.
"""

import argparse
import datetime
import glob
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expiry', type=int, required=True)
    parser.add_argument('--rm', default=False, action='store_true')
    options = parser.parse_args()

    expiry = datetime.datetime.now() - datetime.timedelta(
                seconds=options.expiry)
    sys.stderr.write('Looking for cache unused since {}\n'.format(expiry))
    expiry_ts = expiry.timestamp()

    total_size = 0
    trim_size = 0
    for f in glob.glob(os.path.expanduser('~/.cupy/kernel_cache/*.cubin')):
        stat = os.lstat(f)
        total_size += stat.st_size
        if stat.st_atime < expiry_ts:
            print(f)
            trim_size += stat.st_size
            if options.rm:
                os.unlink(f)
    sys.stderr.write('Total: {} bytes\n'.format(total_size))
    sys.stderr.write('Expired: {} bytes\n'.format(trim_size))


if __name__ == '__main__':
    main()
