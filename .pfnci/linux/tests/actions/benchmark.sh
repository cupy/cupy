#!/bin/bash

set -uex


git clone https://github.com/cupy/cupy-performance.git performance
# TODO(ecastill): make this optional
pip install seaborn
ls
pushd performance
python prof.py benchmarks/bench_ufunc_cupy.py -c
# Compare with current branch
for bench in *.csv
do
    python regression_detect.py ${bench} /perf-results/head/${bench}
done

cp *.csv /perf-results/
popd
