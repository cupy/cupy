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
git checkout master
pip install --user -v .

# Run benchmarks for master
python prof.py benchmarks/bench_ufunc_cupy.py -c
mkdir master
mv *.csv master/

# Compare with current branch
for bench in *.csv
do
    python regresion_detect.py master/${bench} pr/${bench}
done

cp *.csv /perf-results/
popd
