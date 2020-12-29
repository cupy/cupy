# cuSPARSE
python gen_pyx.py directives/cusparse.py templates/cusparse.pyx.template > sample_cusparse.pyx
python gen_pxd.py directives/cusparse.py templates/default.pxd.template > sample_cusparse.pxd

# cuSOLVER
python gen_pyx.py directives/cusolver.py templates/cusolver.pyx.template > sample_cusolver.pyx
python gen_pxd.py directives/cusolver.py templates/cusolver.pxd.template > sample_cusolver.pxd

# cuBLAS
python gen_pyx.py directives/cublas.py templates/cublas.pyx.template > sample_cublas.pyx
python gen_pxd.py directives/cublas.py templates/cublas.pxd.template > sample_cublas.pxd

# cuTENSOR
python gen_pyx.py directives/cutensor.py templates/cutensor.pyx.template > sample_cutensor.pyx
python gen_pxd.py directives/cutensor.py templates/default.pxd.template > sample_cutensor.pxd
