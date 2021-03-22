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
_nccl_records = []
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


def _make_nccl_url(public_version, filename):
    # https://developer.download.nvidia.com/compute/redist/nccl/v2.8/nccl_2.8.4-1+cuda11.2_x86_64.txz
    return (
        'https://developer.download.nvidia.com/compute/redist/nccl/' +
        'v{}/{}'.format(public_version, filename))


def _make_nccl_record(
        cuda_version, full_version, public_version, filename_linux):
    return {
        'cuda': cuda_version,
        'nccl': full_version,
        'assets': {
            'Linux': {
                'url': _make_nccl_url(public_version, filename_linux),
                'filename': 'libnccl.so.{}'.format(full_version),
            },
        },
    }


_nccl_records.append(_make_nccl_record(
    '11.2', '2.8.4', '2.8',
    'nccl_2.8.4-1+cuda11.2_x86_64.txz'))
_nccl_records.append(_make_nccl_record(
    '11.1', '2.8.4', '2.8',
    'nccl_2.8.4-1+cuda11.1_x86_64.txz'))
_nccl_records.append(_make_nccl_record(
    '11.0', '2.8.4', '2.8',
    'nccl_2.8.4-1+cuda11.0_x86_64.txz'))
_nccl_records.append(_make_nccl_record(
    '10.2', '2.8.4', '2.8',
    'nccl_2.8.4-1+cuda10.2_x86_64.txz'))
_nccl_records.append(_make_nccl_record(
    '10.1', '2.8.3', '2.8',
    'nccl_2.8.3-1+cuda10.1_x86_64.txz'))
_nccl_records.append(_make_nccl_record(
    '10.0', '2.6.4', '2.6',
    'nccl_2.6.4-1+cuda10.0_x86_64.txz'))
_nccl_records.append(_make_nccl_record(
    '9.2', '2.4.8', '2.4',
    'nccl_2.4.8-1+cuda9.2_x86_64.txz'))
library_records['nccl'] = _nccl_records


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

    target_platform = platform.system()
    asset = record['assets'].get(target_platform, None)
    if asset is None:
        raise RuntimeError('''
The current platform ({}) is not supported.'''.format(target_platform))

    print('Installing {} {} for CUDA {} to: {}'.format(
        library, record[library], record['cuda'], destination))

    url = asset['url']
    print('Downloading {}...'.format(url))
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, os.path.basename(url)), 'wb') as f:
            with urllib.request.urlopen(url) as response:
                f.write(response.read())
        print('Extracting...')
        outdir = os.path.join(tmpdir, 'extract')
        shutil.unpack_archive(f.name, outdir)
        print('Installing...')
        if library == 'cudnn':
            shutil.move(os.path.join(outdir, 'cuda'), destination)
        elif library == 'cutensor':
            if cuda.startswith('11.'):
                cuda = '11'
            shutil.move(
                os.path.join(outdir, 'libcutensor', 'include'),
                os.path.join(destination, 'include'))
            shutil.move(
                os.path.join(outdir, 'libcutensor', 'lib', cuda),
                os.path.join(destination, 'lib'))
            shutil.move(
                os.path.join(outdir, 'libcutensor', 'license.pdf'),
                destination)
        elif library == 'nccl':
            subdir = os.listdir(outdir)  # ['nccl_2.8.4-1+cuda11.2_x86_64']
            assert len(subdir) == 1
            shutil.move(os.path.join(outdir, subdir[0]), destination)
        else:
            assert False
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
                        choices=['cudnn', 'cutensor', 'nccl'],
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

    if params.action == 'install':
        install_lib(params.cuda, params.prefix, params.library)
    elif params.action == 'dump':
        print(json.dumps(library_records[params.library], indent=4))
    else:
        assert False


if __name__ == '__main__':
    main(sys.argv[1:])
