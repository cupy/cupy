# cuSPARSE
python gen_pyx.py directives/cusparse.py templates/cusparse.pyx.template > sample_cusparse.pyx && \
python gen_pxd.py directives/cusparse.py templates/default.pxd.template > sample_cusparse.pxd && \
python gen_stub.py directives/cusparse.py templates/default.stub.template > sample_cusparse_stub.h && \
python gen_compat.py directives/cusparse.py templates/cupy_cusparse.h.template > sample_cupy_cusparse.h && \

# cuSOLVER
python gen_pyx.py directives/cusolver.py templates/cusolver.pyx.template > sample_cusolver.pyx && \
python gen_pxd.py directives/cusolver.py templates/cusolver.pxd.template > sample_cusolver.pxd && \
python gen_stub.py directives/cusolver.py templates/default.stub.template > sample_cusolver_stub.h && \
python gen_compat.py directives/cusolver.py templates/cupy_cusolver.h.template > sample_cupy_cusolver.h && \

# cuBLAS
python gen_pyx.py directives/cublas.py templates/cublas.pyx.template > sample_cublas.pyx && \
python gen_pxd.py directives/cublas.py templates/cublas.pxd.template > sample_cublas.pxd && \
python gen_stub.py directives/cublas.py templates/default.stub.template > sample_cublas_stub.h  && \
python gen_compat.py directives/cublas.py templates/cupy_cublas.h.template > sample_cupy_cublas.h

# cuTENSOR
# TODO(takagi) Support cuTENSOR?
#python gen_pyx.py directives/cutensor.py templates/cutensor.pyx.template > sample_cutensor.pyx && \
#python gen_pxd.py directives/cutensor.py templates/default.pxd.template > sample_cutensor.pxd && \
#python gen_stub.py directives/cutensor.py templates/default.stub.template > sample_cutensor_stub.h
