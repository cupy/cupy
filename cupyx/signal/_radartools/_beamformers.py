# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import cupy as cp


def mvdr(x, sv, calc_cov=True):
    """
    Minimum variance distortionless response (MVDR) beamformer weights

    Parameters
    ----------
    x : ndarray
        Received signal or input covariance matrix, assume 2D array with
        size [num_sensors, num_samples]

    sv: ndarray
        Steering vector, assume 1D array with size [num_sensors, 1]

    calc_cov : bool
        Determine whether to calculate covariance matrix. Simply put, calc_cov
        defines whether x input is made of sensor/observation data or is
        a precalculated covariance matrix

    Note: Unlike MATLAB where input matrix x is of size MxN where N represents
    the number of array elements, we assume row-major formatted data where each
    row is assumed to be complex-valued data from a given sensor (i.e. NxM)
    """
    if x.shape[0] > x.shape[1]:
        raise ValueError(
            "Matrix has more sensors than samples. Consider \
            transposing and remember cuSignal is row-major, unlike MATLAB"
        )

    if x.shape[0] != sv.shape[0]:
        raise ValueError("Steering Vector and input data do not align")

    if calc_cov:
        R = cp.cov(x)
    else:
        R = cp.asarray(x)

    R_inv = cp.linalg.inv(R)
    svh = cp.transpose(cp.conj(sv))

    wB = cp.matmul(R_inv, sv)
    # wA is a 1x1 scalar
    wA = cp.matmul(svh, wB)
    w = wB / wA

    return w
