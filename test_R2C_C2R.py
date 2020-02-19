import cupy as cp
import numpy as np


#a = cp.random.random((2,3,7))
#planR2C = cp.cuda.cufft.PlanNd((2,3,7), [2,3,7], 1, 2*3*7, 
#                                        [2,3,4], 1, 2*3*4, 
#                                        cp.cuda.cufft.CUFFT_D2Z, 1, 'C', 2)
#
#b_np = np.fft.rfftn(cp.asnumpy(a))
#b_planR2C = cp.zeros((2,3,4), dtype=cp.complex128, order='c')
#planR2C.fft(a, b_planR2C, 0)
##print(b_planR2C)
#print(cp.allclose(b_np, b_planR2C))
#
#b_cp = cp.fft.rfftn(a)
#print(cp.allclose(b_np, b_cp))


a = cp.random.random((2,4,8)).astype(cp.float32)
planR2C = cp.cuda.cufft.PlanNd((2,4,8), #[2,4,8], 1, 2*4*8, 
                                        #[2,4,5], 1, 2*4*5, 
                                        None, 1, 0,
                                        None, 1, 0,
                                        cp.cuda.cufft.CUFFT_R2C, 1, 'C', 2)

b_np = np.fft.rfftn(cp.asnumpy(a))
b_planR2C = cp.zeros((2,4,5), dtype=cp.complex64, order='c')
planR2C.fft(a, b_planR2C, 0)
#print(b_planR2C)
print(cp.allclose(b_np, b_planR2C))

b_cp = cp.fft.rfftn(a)
print(cp.allclose(b_np, b_cp))

print(b_planR2C)

planC2R = cp.cuda.cufft.PlanNd((2,4,8), None, 1, 0,
                                        None, 1, 0,
                                        #[2,4,8], 1, 2*4*8,
                                        #[2,4,8], 1, 2*4*8,
                                        cp.cuda.cufft.CUFFT_C2R, 1, 'C', 2)
b_planC2R = cp.zeros((2,4,20), dtype=cp.float32, order='c')
#print(b_planR2C.dtype)
planR2C.fft(b_planR2C, b_planC2R, 0)
print(cp.allclose(a, b_planC2R[..., :8]))
print(a, "\n\n ***** \n\n", b_planC2R)

cp.fft.config.enable_nd_planning = False
b_final = cp.fft.irfftn(b_planR2C)
print(cp.allclose(a, b_final))


#a = cp.random.random((2,3,4)) + 1j*cp.random.random((2,3,4))
##a_masked = cp.zeros((2,3,6), dtype=cp.complex128, order='c')
##a_masked[..., :4] = a
#for i in range(1):
#    for j in range(2):
#        for k in range(2):
#            a[i, j, k] = a[1-i, 2-j, 3-k].conj()
#planC2R = cp.cuda.cufft.PlanNd((2,3,4), None, 1, 0, 
#                                        None, 1, 0, 
#                                        cp.cuda.cufft.CUFFT_Z2D, 1, 'C', 2)
#
#b_np = np.fft.irfftn(cp.asnumpy(a))
#print((b_np.shape))
#b_planC2R = cp.zeros((2,3,6), dtype=cp.float64, order='c')
#planC2R.fft(a, b_planC2R, 0)
##b_planC2R /= 36.
#print(cp.allclose(b_np, b_planC2R))
#print(b_np, "\n\n ***** \n\n", b_planC2R)
#
#cp.fft.config.enable_nd_planning = False
#b_cp = cp.fft.irfftn(a)
#print(cp.allclose(b_np, b_cp))


#a = cp.random.random((2,3,7))
#planR2C = cp.cuda.cufft.PlanNd((2,3,5), [2,3,5], 1, 2*3*5, 
#                                        [2,3,3], 1, 2*3*3, 
#                                        cp.cuda.cufft.CUFFT_D2Z, 1, 'C', 2)
#
#b_np = np.fft.rfftn(cp.asnumpy(a), s=(2,3,5))
#print(b_np.shape)
#b_planR2C = cp.zeros((2,3,3), dtype=cp.complex128, order='c')
#a_masked = cp.zeros((2,3,5), dtype=a.dtype, order='c')
#a_masked[..., 0:5] = a[..., 0:5] 
#planR2C.fft(a_masked, b_planR2C, 0)
#print(cp.allclose(b_np, b_planR2C))
##print(b_np, "\n\n ***** \n\n", b_planR2C)
#
#b_cp = cp.fft.rfftn(a, s=(2,3,5))
#print(cp.allclose(b_np, b_cp))
