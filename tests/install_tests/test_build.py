from __future__ import annotations

from distutils import ccompiler
from distutils import sysconfig
import os
import unittest
from unittest import mock

import pytest

from cupy_builder._context import Context
from cupy_builder import install_build as build


test_hip = bool(int(os.environ.get('CUPY_INSTALL_USE_HIP', '0')))


class TestCheckVersion(unittest.TestCase):

    def setUp(self):
        ctx = Context('.', _env={}, _argv=[])
        self.compiler = ccompiler.new_compiler()
        sysconfig.customize_compiler(self.compiler)
        self.settings = build.get_compiler_setting(ctx, test_hip)

    @pytest.mark.skipif(not test_hip, reason='For ROCm/HIP environment')
    def test_check_hip_version(self):
        with self.assertRaises(RuntimeError):
            build.get_hip_version()
        assert build.check_hip_version(
            self.compiler, self.settings)
        assert isinstance(build.get_hip_version(), int)
        assert isinstance(build.get_hip_version(True), str)

    def test_hiptensor_minimum_version(self):
        assert not build._is_supported_hiptensor_version(0)
        assert not build._is_supported_hiptensor_version(2_002_999)
        assert build._is_supported_hiptensor_version(2_003_000)

    def test_check_hiptensor_minimum_version(self):
        for version, expected in (
                ('0', False),
                ('2002999', False),
                ('2003000', True)):
            with self.subTest(version=version):
                with mock.patch.object(
                        build, 'build_and_run', return_value=version):
                    assert build.check_hiptensor_version(
                        self.compiler, self.settings) is expected
