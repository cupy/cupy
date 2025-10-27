#!/usr/bin/env python

"""
CUDA Library Installer

Installs the latest CUDA library supported by CuPy.
"""

# This script will also be used as a standalone script when building wheels.
# Keep the script runnable without CuPy dependency.
from __future__ import annotations


import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request


_deprecation_message = """
******************************************************************************

The "cupyx.tools.install_library" tool is deprecated and will be removed in
a future CuPy release. To install NCCL/cuTENSOR libraries, please install them
via package managers (pip/conda) that are used to install CuPy.

You can use the following command to see if NCCL/cuTENSOR is already installed.
If you see the version number displayed for "NCCL Runtime Version" or "cuTENSOR
Version", they are available and enabled in your CuPy installation.

  $ python -c 'import cupy; cupy.show_config(_full=True)'

If you see "None" instead of the version number, and you installed CuPy via pip
or conda, you can get the instructions to install these libraries by running
the following commands:

  $ python -c 'import cupy.cuda.nccl'
  $ python -c 'import cupy.cuda.cutensor'

******************************************************************************
"""

_cutensor_records = []
_nccl_records = []
library_records = {}


def _make_cutensor_url(platform, filename):
    # https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/windows-x86_64/libcutensor-windows-x86_64-2.3.1.0_cuda13-archive.zip
    return (
        'https://developer.download.nvidia.com/compute/cutensor/' +
        f'redist/libcutensor/{platform}-x86_64/{filename}')


def __make_cutensor_record(
        cuda_version, public_version, min_pypi_version,
        filename_linux, filename_windows):
    return {
        'cuda': cuda_version,
        'cutensor': public_version,
        'min_pypi_version': min_pypi_version,
        'assets': {
            'Linux:x86_64': {
                'url': _make_cutensor_url('linux', filename_linux),
                'filenames': [
                    'libcutensor.so.{}'.format(public_version),
                    'libcutensorMg.so.{}'.format(public_version),
                ],
            },
            'Windows:x86_64': {
                'url': _make_cutensor_url('windows', filename_windows),
                'filenames': ['cutensor.dll', 'cutensorMg.dll'],
            },
        }
    }


def _make_cutensor_record(cuda_version):
    # cuTENSOR guarantees ABI compatibility within the major version (#9017).
    # `min_pypi_version` must be bumped only when:
    # (1) Bumping the major version, or
    # (2) CuPy started to use APIs introduced in minor versions
    cuda_major = cuda_version.split('.')[0]
    return __make_cutensor_record(
        cuda_version, '2.3.0', '2.3.0',
        f'libcutensor-linux-x86_64-2.3.1.0_cuda{cuda_major}-archive.tar.xz',
        f'libcutensor-windows-x86_64-2.3.1.0_cuda{cuda_major}-archive.zip',
    )


_cutensor_records.append(_make_cutensor_record('12.x'))
_cutensor_records.append(_make_cutensor_record('13.x'))
library_records['cutensor'] = _cutensor_records


def _make_nccl_url(public_version, filename):
    # https://developer.download.nvidia.com/compute/redist/nccl/v2.8/nccl_2.8.4-1+cuda11.2_x86_64.txz
    return (
        'https://developer.download.nvidia.com/compute/redist/nccl/' +
        f'v{public_version}/{filename}')


def _make_nccl_record(
        cuda_version, full_version, public_version, min_pypi_version,
        filename_linux_x86_64, filename_linux_aarch64):
    return {
        'cuda': cuda_version,
        'nccl': full_version,
        'min_pypi_version': min_pypi_version,
        'assets': {
            'Linux:x86_64': {
                'url': _make_nccl_url(
                    public_version, filename_linux_x86_64),
                'filenames': ['libnccl.so.{}'.format(full_version)],
            },
            'Linux:aarch64': {
                'url': _make_nccl_url(
                    public_version, filename_linux_aarch64),
                'filenames': ['libnccl.so.{}'.format(full_version)],
            },
        },
    }


# https://docs.nvidia.com/deeplearning/nccl/release-notes/overview.html
_nccl_records.append(_make_nccl_record(
    '13.x', '2.27.7', '2.27.7', '2.27.7',
    'nccl_2.27.7-1+cuda13.0_x86_64.txz',
    'nccl_2.27.7-1+cuda13.0_aarch64.txz'))
_nccl_records.append(_make_nccl_record(
    '12.x', '2.25.1', '2.25.1', '2.16.5',
    'nccl_2.25.1-1+cuda12.8_x86_64.txz',
    'nccl_2.25.1-1+cuda12.8_aarch64.txz'))
_nccl_records.append(_make_nccl_record(
    '11.x', '2.16.5', '2.16.5', '2.16.5',  # CUDA 11.2+
    'nccl_2.16.5-1+cuda11.8_x86_64.txz',
    'nccl_2.16.5-1+cuda11.8_aarch64.txz'))
library_records['nccl'] = _nccl_records


def _unpack_archive(filename, extract_dir):
    try:
        shutil.unpack_archive(filename, extract_dir)
    except shutil.ReadError:
        print('The archive format is not supported in your Python '
              'environment. Falling back to "tar" command...')
        try:
            os.makedirs(extract_dir, exist_ok=True)
            subprocess.run(
                ['tar', 'xf', filename, '-C', extract_dir], check=True)
        except subprocess.CalledProcessError:
            msg = 'Failed to extract the archive using "tar" command.'
            raise RuntimeError(msg)


def install_lib(cuda, prefix, library, arch):
    if library == 'nccl' and arch in ('x86_64', 'aarch64'):
        pass  # Supported
    elif arch != 'x86_64':
        raise RuntimeError('''
Currently this tool only supports x86_64 or aarch64 architecture for NCCL.''')
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

    target_platform = f'{platform.system()}:{arch}'
    asset = record['assets'].get(target_platform, None)
    if asset is None:
        raise RuntimeError('''
The current platform ({}) is not supported.'''.format(target_platform))

    if library == 'cutensor':
        print('By downloading and using cuTENSOR, you accept the terms and'
              ' conditions of the NVIDIA cuTENSOR Software License Agreement:')
        print('  https://docs.nvidia.com/cuda/cutensor/license.html')
        print()
    elif library == 'nccl':
        pass  # BSD
    else:
        assert False

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
        _unpack_archive(f.name, outdir)

        subdir = os.listdir(outdir)
        assert len(subdir) == 1
        dir_name = subdir[0]

        print('Installing...')
        if library == 'cutensor':
            license = 'LICENSE'
            shutil.move(
                os.path.join(outdir, dir_name, 'include'),
                os.path.join(destination, 'include'))
            shutil.move(
                os.path.join(outdir, dir_name, 'lib'),
                os.path.join(destination, 'lib'))
            shutil.move(
                os.path.join(outdir, dir_name, license), destination)
        elif library == 'nccl':
            shutil.move(os.path.join(outdir, dir_name), destination)
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

    parser.add_argument('--library',
                        choices=['cutensor', 'nccl'],
                        required=True,
                        help='Library to install')
    parser.add_argument('--cuda', type=str, required=True,
                        help='CUDA version')
    parser.add_argument('--arch', choices=['x86_64', 'aarch64'],
                        default=None,
                        help='Target arhitecture (x86_64 or aarch64)')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Install destination')
    parser.add_argument('--action', choices=['install', 'dump'],
                        default='install',
                        help='Action to perform')
    params = parser.parse_args(args)

    if params.arch is None:
        machine = platform.uname().machine.lower()
        if machine in ('x86_64', 'amd64'):
            params.arch = 'x86_64'
        elif machine == 'aarch64':
            params.arch = 'aarch64'
        else:
            raise AssertionError(f'unsupported architecture: {machine}')

    if params.prefix is not None:
        params.prefix = os.path.abspath(params.prefix)

    if params.action == 'install':
        print(_deprecation_message)
        install_lib(params.cuda, params.prefix, params.library,
                    params.arch)
    elif params.action == 'dump':
        # This option is only for internal use by cupy-release-tools.
        print(json.dumps(library_records[params.library], indent=4))
    else:
        assert False


if __name__ == '__main__':
    main(sys.argv[1:])
