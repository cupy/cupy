#!/usr/bin/env python

"""
CUDA Library Installer

Installs the latest CUDA library supported by CuPy.
"""

# This script will also be used as a standalone script when building wheels.
# Keep the script runnable without CuPy dependency.

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import urllib.request


_cudnn_records = []


def _make_cudnn_url(public_version, filename):
    # https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.2/cudnn-11.0-linux-x64-v8.0.2.39.tgz
    return (
        'https://developer.download.nvidia.com/compute/redist/cudnn' +
        '/v{}/{}'.format(public_version, filename))


def _make_cudnn_record(
        cuda_version, public_version, filename_linux, filename_windows):
    major_version = public_version.split('.')[0]
    return {
        'cuda': cuda_version,
        'cudnn': public_version,
        'assets': {
            'Linux': {
                'url': _make_cudnn_url(public_version, filename_linux),
                'filename': 'libcudnn.so.{}'.format(public_version),
            },
            'Windows': {
                'url': _make_cudnn_url(public_version, filename_windows),
                'filename': 'cudnn64_{}.dll'.format(major_version),
            },
        }
    }


# Latest cuDNN versions: https://developer.nvidia.com/rdp/cudnn-download
_cudnn_records.append(_make_cudnn_record(
    '11.2', '8.1.1',
    'cudnn-11.2-linux-x64-v8.1.1.33.tgz',
    'cudnn-11.2-windows-x64-v8.1.1.33.zip'))
_cudnn_records.append(_make_cudnn_record(
    '11.1', '8.1.1',
    'cudnn-11.2-linux-x64-v8.1.1.33.tgz',
    'cudnn-11.2-windows-x64-v8.1.1.33.zip'))
_cudnn_records.append(_make_cudnn_record(
    '11.0', '8.1.1',
    'cudnn-11.2-linux-x64-v8.1.1.33.tgz',
    'cudnn-11.2-windows-x64-v8.1.1.33.zip'))
_cudnn_records.append(_make_cudnn_record(
    '10.2', '8.1.1',
    'cudnn-10.2-linux-x64-v8.1.1.33.tgz',
    'cudnn-10.2-windows10-x64-v8.1.1.33.zip'))
_cudnn_records.append(_make_cudnn_record(
    '10.1', '8.0.5',
    'cudnn-10.1-linux-x64-v8.0.5.39.tgz',
    'cudnn-10.1-windows10-x64-v8.0.5.39.zip'))
_cudnn_records.append(_make_cudnn_record(
    '10.0', '7.6.5',
    'cudnn-10.0-linux-x64-v7.6.5.32.tgz',
    'cudnn-10.0-windows10-x64-v7.6.5.32.zip'))
_cudnn_records.append(_make_cudnn_record(
    '9.2', '7.6.5',
    'cudnn-9.2-linux-x64-v7.6.5.32.tgz',
    'cudnn-9.2-windows10-x64-v7.6.5.32.zip'))
_cudnn_records.append(_make_cudnn_record(
    '9.0', '7.6.5',
    'cudnn-9.0-linux-x64-v7.6.5.32.tgz',
    'cudnn-9.0-windows10-x64-v7.6.5.32.zip'))


def install_cudnn(cuda, prefix):
    record = None
    for record in _cudnn_records:
        if record['cuda'] == cuda:
            break
    else:
        raise RuntimeError('''
The CUDA version specified is not supported.
Should be one of {}.'''.format(str([x['cuda'] for x in _cudnn_records])))
    if prefix is None:
        prefix = os.path.expanduser('~/.cupy/cuda_lib')
    destination = calculate_destination(prefix, cuda, 'cudnn', record['cudnn'])

    if os.path.exists(destination):
        raise RuntimeError('''
The destination directory {} already exists.
Remove the directory first if you want to reinstall.'''.format(destination))
    asset = record['assets'][platform.system()]

    print('Installing cuDNN {} for CUDA {} to: {}'.format(
        record['cudnn'], record['cuda'], destination))

    url = asset['url']
    print('Downloading {}...'.format(url))
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, os.path.basename(url)), 'wb') as f:
            with urllib.request.urlopen(url) as response:
                f.write(response.read())
        print('Extracting...')
        shutil.unpack_archive(f.name, tmpdir)
        print('Installing...')
        shutil.move(os.path.join(tmpdir, 'cuda'), destination)
        print('Cleaning up...')
    print('Done!')


def calculate_destination(prefix, cuda, lib, lib_ver):
    """Calculates the installation directory.

    ~/.cupy/cuda_lib/{cuda_version}/{library_name}/{library_version}
    """
    return os.path.join(prefix, cuda, lib, lib_ver)


def main(args):
    parser = argparse.ArgumentParser()

    # TODO(kmaehashi) support cuTENSOR and NCCL
    parser.add_argument('--library', choices=['cudnn'], required=True,
                        help='Library to install')
    parser.add_argument('--cuda', type=str, required=True,
                        help='CUDA version')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Install destination')
    parser.add_argument('--action', choices=['install', 'dump'],
                        default='install',
                        help='Action to perform')
    params = parser.parse_args(args)

    if params.prefix is not None:
        params.prefix = os.path.abspath(params.prefix)

    if params.library == 'cudnn':
        if params.action == 'install':
            install_cudnn(params.cuda, params.prefix)
        elif params.action == 'dump':
            print(json.dumps(_cudnn_records, indent=4))
        else:
            assert False
    else:
        assert False


if __name__ == '__main__':
    main(sys.argv[1:])
