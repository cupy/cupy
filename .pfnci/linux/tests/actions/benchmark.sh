#!/bin/bash

set -uex


git clone https://github.com/cupy/cupy-performance.git performance
# TODO(ecastill): make this optional
pip install seaborn
ls
pushd performance
python prof.py benchmarks/bench_ufunc_cupy.py -c

mkdir pr
mv *.csv pr/

pip uninstall -y cupy
pip install cupy-cuda115

# Run benchmarks for master
python prof.py benchmarks/bench_ufunc_cupy.py -c
mkdir master
mv *.csv master/

# Compare with current branch
for bench in master/*.csv
do
    # python regresion_detect.py /perf-results/head/${bench} ${bench}
    python regresion_detect.py master/${bench} pr/${bench}
done

cp *.csv /perf-results/
popd
