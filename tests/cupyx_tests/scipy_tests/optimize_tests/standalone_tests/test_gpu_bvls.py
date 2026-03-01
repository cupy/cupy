#!/usr/bin/python
import sys
import numpy as np
import cupy as cp
from scipy.optimize import lsq_linear
from cupyx.scipy.optimize import gpu_lsq_linear
import time


"""Stand-alone test program to test timing and accuracy of GPU
implementation of BVLS"""

d = 1000
m = 7
n = 7

if len(sys.argv) > 1 and sys.argv[1] == '-h':
    print ("Usage: test_gpu_bvls.py [d] [d m n]")
    print ("Where dimensions of A are (d,m,n) and b are (d,m)")
    print ("Either include all 3 dimensions or only d")
elif len(sys.argv) > 1:
    d = int(sys.argv[1])

if len(sys.argv) > 3:
    m = int(sys.argv[2])
    n = int(sys.argv[3])


rng = np.random.default_rng()
A = np.random.random((d,m,n))
b = rng.standard_normal((d,m))
lb = rng.standard_normal(n)
ub = lb+1

gbounds = cp.asarray((cp.asarray(lb), cp.asarray(ub)))
gb = cp.asarray(b)
ga = cp.asarray(A)

#prime GPU before test
gres = gpu_lsq_linear(ga[:2], gb[:2], bounds=gbounds, method='bvls')

cx = np.zeros((d,n))
print (f'Timing CPU implementation of solving {d=} arrays Ax = b where A has dimension {m=}, {n=}')
t = time.time()
for j in range(d):
    res = lsq_linear(A[j], b[j], bounds=(lb,ub), method='bvls')
    cx[j] = res.x
tcpu = time.time()-t
print(f'\tCPU time = {tcpu=}')

print (f'Timing GPU implementation of solving {d=} arrays Ax = b where A has dimension {m=}, {n=}')
t = time.time()
gres = gpu_lsq_linear(ga, gb, bounds=gbounds, method='bvls')
tgpu = time.time()-t
print(f'\tGPU time = {tgpu=}')

print (f'Timing GPU without returning cost and optimality')
t = time.time()
gres = gpu_lsq_linear(ga, gb, bounds=gbounds, method='bvls', return_optimality=False, return_cost=False)
tgpu = time.time()-t
print(f'\tGPU time = {tgpu=}')
gx = gres.x.get()

isClose = np.allclose(gx, cx)
std = (gx-cx).std()
print(f'Resuls: {isClose=}, {std=}')
