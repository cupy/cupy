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
_cutensor_records = []
library_records = {}


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
    '11.2', '8.1.0',
    'cudnn-11.2-linux-x64-v8.1.0.77.tgz',
    'cudnn-11.2-windows-x64-v8.1.0.77.zip'))
_cudnn_records.append(_make_cudnn_record(
    '11.1', '8.1.0',
    'cudnn-11.2-linux-x64-v8.1.0.77.tgz',
    'cudnn-11.2-windows-x64-v8.1.0.77.zip'))
_cudnn_records.append(_make_cudnn_record(
    '11.0', '8.1.0',
    'cudnn-11.2-linux-x64-v8.1.0.77.tgz',
    'cudnn-11.2-windows-x64-v8.1.0.77.zip'))
_cudnn_records.append(_make_cudnn_record(
    '10.2', '8.1.0',
    'cudnn-10.2-linux-x64-v8.1.0.77.tgz',
    'cudnn-10.2-windows10-x64-v8.1.0.77.zip'))
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
library_records['cudnn'] = _cudnn_records


def _make_cutensor_url(public_version, filename):
    # https://developer.download.nvidia.com/compute/cutensor/1.2.2/local_installers/libcutensor-linux-x86_64-1.2.2.5.tar.gz
    return (
        'https://developer.download.nvidia.com/compute/cutensor/' +
        '{}/local_installers/{}'.format(public_version, filename))


def _make_cutensor_record(
        cuda_version, public_version, filename_linux, filename_windows):
    return {
        'cuda': cuda_version,
        'cutensor': public_version,
        'assets': {
            'Linux': {
                'url': _make_cutensor_url(public_version, filename_linux),
                'filename': 'libcutensor.so.{}'.format(public_version),
            },
            'Windows': {
                'url': _make_cutensor_url(public_version, filename_windows),
                'filename': 'cutensor.dll',
            },
        }
    }


_cutensor_records.append(_make_cutensor_record(
    '11.2', '1.2.2',
    'libcutensor-linux-x86_64-1.2.2.5.tar.gz',
    'libcutensor-windows-x86_64-1.2.2.5.zip'))
_cutensor_records.append(_make_cutensor_record(
    '11.1', '1.2.2',
    'libcutensor-linux-x86_64-1.2.2.5.tar.gz',
    'libcutensor-windows-x86_64-1.2.2.5.zip'))
_cutensor_records.append(_make_cutensor_record(
    '11.0', '1.2.2',
    'libcutensor-linux-x86_64-1.2.2.5.tar.gz',
    'libcutensor-windows-x86_64-1.2.2.5.zip'))
_cutensor_records.append(_make_cutensor_record(
    '10.2', '1.2.2',
    'libcutensor-linux-x86_64-1.2.2.5.tar.gz',
    'libcutensor-windows-x86_64-1.2.2.5.zip'))
_cutensor_records.append(_make_cutensor_record(
    '10.1', '1.2.2',
    'libcutensor-linux-x86_64-1.2.2.5.tar.gz',
    'libcutensor-windows-x86_64-1.2.2.5.zip'))
library_records['cutensor'] = _cutensor_records


def install_lib(cuda, prefix, library):
    record = None
    lib_records = library_records
    for record in lib_records[library]:
        if record['cuda'] == cuda:
            break
    else:
        raise RuntimeError('''
The CUDA version specified is not supported.
Should be one of {}.'''.format(str([x['cuda'] for x in lib_records[library]])))
    if prefix is None:
        prefix = os.path.expanduser('~/.cupy/cuda_lib')
    destination = calculate_destination(prefix, cuda, library, record[library])

    if os.path.exists(destination):
        raise RuntimeError('''
The destination directory {} already exists.
Remove the directory first if you want to reinstall.'''.format(destination))
    asset = record['assets'][platform.system()]

    print('Installing {} {} for CUDA {} to: {}'.format(
        library, record[library], record['cuda'], destination))

    url = asset['url']
    print('Downloading {}...'.format(url))
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, os.path.basename(url)), 'wb') as f:
            with urllib.request.urlopen(url) as response:
                f.write(response.read())
        print('Extracting...')
        shutil.unpack_archive(f.name, tmpdir)
        print('Installing...')
        if library == 'cudnn':
            shutil.move(os.path.join(tmpdir, 'cuda'), destination)
        elif library == 'cutensor':
            include = os.path.join(destination, 'include')
            lib = os.path.join(destination, 'lib64')
            shutil.move(os.path.join(tmpdir, 'libcutensor/include'), include)
            if cuda.startswith('11'):
                cuda = '11'
            shutil.move(os.path.join(tmpdir, 'libcutensor/lib', cuda), lib)
        print('Cleaning up...')
    print('Done!')


def calculate_destination(prefix, cuda, lib, lib_ver):
    """Calculates the installation directory.

    ~/.cupy/cuda_lib/{cuda_version}/{library_name}/{library_version}
    """
    return os.path.join(prefix, cuda, lib, lib_ver)


def main(args):
    parser = argparse.ArgumentParser()

    # TODO(kmaehashi): support NCCL
    parser.add_argument('--library',
                        choices=['cudnn', 'cutensor'],
                        required=True,
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
            install_lib(params.cuda, params.prefix, 'cudnn')
        elif params.action == 'dump':
            print(json.dumps(_cudnn_records, indent=4))
        else:
            assert False
    elif params.library == 'cutensor':
        if params.action == 'install':
            install_lib(params.cuda, params.prefix, 'cutensor')
        elif params.action == 'dump':
            print(json.dumps(_cutensor_records, indent=4))
        else:
            assert False
    else:
        assert False


if __name__ == '__main__':
    main(sys.argv[1:])
