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

import cupy
from cupyx.signal._estimation import _filters_cuda
from cupyx.signal._utils import _helper_tools


class KalmanFilter:
    """
    This is a multi-point Kalman Filter implementation of
    https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/kalman_filter.py,
    with a subset of functionality. Also see [2]_.

    All Kalman Filter matrices are stack on the X axis. This is to allow
    for optimal global accesses on the GPU.

    Parameters
    ----------
    dim_x : int
        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
        This is used to set the default size of P, Q, and u

    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.

    dim_u : int (optional)
        Size of the control input, if it is being used.
        Default value of 0 indicates it is not used.

    points : int (optional)
        Number of Kalman Filter points to track.

    dtype : dtype (optional)
        Data type of compute.

    Attributes
    ----------
    x : array(points, dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.

    P : array(points, dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.

    z : array(points, dim_z, 1)
        Last measurement used in update(). Read only.

    R : array(points, dim_z, dim_z)
        Measurement noise matrix

    Q : array(points, dim_x, dim_x)
        Process noise matrix

    F : array(points, dim_x, dim_x)
        State Transition matrix

    H : array(points, dim_z, dim_x)
        Measurement function

    _alpha_sq : float (points, 1, 1)
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.

    References
    ----------

    .. [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
       p. 208-212. (2006)

    .. [2] Roger Labbe. "Kalman and Bayesian Filters in Python"
       https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(
        self,
        dim_x,
        dim_z,
        dim_u=0,
        points=1,
        dtype=cupy.float32,
    ):
        self.points = points

        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # Create data arrays
        self.x = cupy.zeros(
            (
                self.points,
                dim_x,
                1,
            ),
            dtype=dtype,
        )  # state

        self.P = cupy.repeat(
            cupy.identity(dim_x, dtype=dtype)[cupy.newaxis, :, :],
            self.points,
            axis=0,
        )  # uncertainty covariance

        self.Q = cupy.repeat(
            cupy.identity(dim_x, dtype=dtype)[cupy.newaxis, :, :],
            self.points,
            axis=0,
        )  # process uncertainty

        self.B = None  # control transition matrix

        self.F = cupy.repeat(
            cupy.identity(dim_x, dtype=dtype)[cupy.newaxis, :, :],
            self.points,
            axis=0,
        )  # state transition matrix

        self.H = cupy.zeros(
            (
                self.points,
                dim_z,
                dim_z,
            ),
            dtype=dtype,
        )  # Measurement function

        self.R = cupy.repeat(
            cupy.identity(dim_z, dtype=dtype)[cupy.newaxis, :, :],
            self.points,
            axis=0,
        )  # process uncertainty

        self._alpha_sq = cupy.ones(
            (
                self.points,
                1,
                1,
            ),
            dtype=dtype,
        )  # fading memory control

        self.z = cupy.empty(
            (
                self.points,
                dim_z,
                1,
            ),
            dtype=dtype,
        )

        # Allocate GPU resources
        numSM = _helper_tools._get_numSM()
        threads_z_axis = 16
        threadsperblock = (self.dim_x, self.dim_x, threads_z_axis)
        blockspergrid = (1, 1, numSM * 20)

        max_threads_per_block = self.dim_x * self.dim_x * threads_z_axis

        # Only need to populate cache once
        # At class initialization
        _filters_cuda._populate_kernel_cache(
            self.x.dtype,
            threads_z_axis,
            self.dim_x,
            self.dim_z,
            self.dim_u,
            max_threads_per_block,
        )

        # Retrieve kernel from cache
        self.predict_kernel = _filters_cuda._get_backend_kernel(
            self.x.dtype,
            blockspergrid,
            threadsperblock,
            "predict",
        )

        self.update_kernel = _filters_cuda._get_backend_kernel(
            self.x.dtype,
            blockspergrid,
            threadsperblock,
            "update",
        )

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        """
        # B will be ignored until implemented
        if u is not None:
            raise NotImplementedError(
                "Control Matrix implementation in process")

        if B is None:
            B = self.B
        else:
            B = cupy.asarray(B)

        if F is None:
            F = self.F
        else:
            F = cupy.asarray(F)

        if Q is None:
            Q = self.Q
        elif cupy.isscalar(Q):
            Q = cupy.repeat(
                (cupy.identity(self.dim_x, dtype=self.x.dtype)
                 * Q)[cupy.newaxis, :, :],
                self.points,
                axis=0,
            )
        else:
            Q = cupy.asarray(Q)

        self.predict_kernel(
            self._alpha_sq,
            self.x,
            u,
            B,
            F,
            self.P,
            Q,
        )

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.

        Parameters
        ----------
        z : array(points, dim_z, 1)
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
            If you pass in a value of H, z must be a column vector the
            of the correct size.

        R : array(points, dim_z, dim_z), scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : array(points, dim_z, dim_x), or None

            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """
        if z is None:
            return

        if R is None:
            R = self.R
        elif cupy.isscalar(R):
            R = cupy.repeat(
                (cupy.identity(self.dim_z, dtype=self.x.dtype)
                 * R)[cupy.newaxis, :, :],
                self.points,
                axis=0,
            )
        else:
            R = cupy.asarray(R)

        if H is None:
            H = self.H
        else:
            H = cupy.asarray(H)

        z = cupy.asarray(z)

        self.update_kernel(
            self.x,
            z,
            H,
            self.P,
            R,
        )
