import filterpy.kalman
import numpy
import pytest

import cupy
from cupy import testing
import cupyx.signal


@pytest.mark.parametrize('dtype', [cupy.float32, cupy.float64])
def test_kalman_filter(dtype):
    # Borrowed from cuSignal:
    # cusignal/notebooks/api_guide/estimation_examples.ipynb

    dim_x = 4
    dim_z = 2
    iterations = 10
    tracks = 32
    dt = dtype

    # State transition matrix
    F = numpy.array(
        [
            [1.0, 0.0, 1.0, 0.0],  # x = x0 + v_x*dt
            [0.0, 1.0, 0.0, 1.0],  # y = y0 + v_y*dt
            [0.0, 0.0, 1.0, 0.0],  # dx = v_x
            [1.0, 0.0, 0.0, 1.0],
        ],  # dy = v_y
        dtype=dt,
    )

    # Measurement function
    H = numpy.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0]
        ],
        dtype=dt,  # x_0  # y_0
    )

    # Initial location
    x = numpy.array([[10.0, 10.0, 0.0, 0.0]], dtype=dt).T  # x, y, v_x, v_y

    # Initial estimate error
    P = numpy.eye(dim_x, dtype=dt) * \
        numpy.array([1.0, 1.0, 2.0, 2.0], dtype=dt)

    # Measurement noise
    R = numpy.eye(dim_z, dtype=dt) * 0.01

    # Motion noise
    Q = numpy.eye(dim_x, dtype=dt) * \
        numpy.array([10.0, 10.0, 10.0, 10.0], dtype=dt)

    # Process CPU Kalman filter
    f_fpy = filterpy.kalman.KalmanFilter(dim_x=dim_x, dim_z=dim_z)

    for _ in range(tracks):
        numpy.random.seed(1234)
        f_fpy.x = x
        f_fpy.F = F
        f_fpy.H = H
        f_fpy.P = P
        f_fpy.R = R
        f_fpy.Q = Q

        for _ in range(iterations):
            f_fpy.predict()
            z = numpy.random.random_sample(dim_z).astype(dt).T
            f_fpy.update(z)

    # Process GPU Kalman filter
    cuS = cupyx.signal.KalmanFilter(dim_x, dim_z, points=tracks, dtype=dt)

    cuS.x = cupy.repeat(cupy.asarray(x[cupy.newaxis, :, :]), tracks, axis=0)
    cuS.F = cupy.repeat(cupy.asarray(F[cupy.newaxis, :, :]), tracks, axis=0)
    cuS.H = cupy.repeat(cupy.asarray(H[cupy.newaxis, :, :]), tracks, axis=0)
    cuS.P = cupy.repeat(cupy.asarray(P[cupy.newaxis, :, :]), tracks, axis=0)
    cuS.R = cupy.repeat(cupy.asarray(R[cupy.newaxis, :, :]), tracks, axis=0)
    cuS.Q = cupy.repeat(cupy.asarray(Q[cupy.newaxis, :, :]), tracks, axis=0)

    numpy.random.seed(1234)

    for _ in range(iterations):
        cuS.predict()

        # ここ np でいいの？
        z = numpy.atleast_2d(numpy.random.random_sample(dim_z).astype(dt)).T
        z = numpy.repeat(z[numpy.newaxis, :, :], tracks, axis=0)

        cuS.update(z)

    # Test results
    testing.assert_allclose(f_fpy.x, cuS.x[0, :, :], rtol=1e-6)
    testing.assert_allclose(f_fpy.x, cuS.x[-1, :, :], rtol=1e-6)

    testing.assert_allclose(f_fpy.P, cuS.P[0, :, :], rtol=1e-6)
    testing.assert_allclose(f_fpy.P, cuS.P[-1, :, :], rtol=1e-6)
