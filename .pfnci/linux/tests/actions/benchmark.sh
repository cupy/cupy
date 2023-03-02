#!/bin/bash

set -uex

git clone https://github.com/cupy/cupy-performance.git performance
# TODO(ecastill): make this optional
python3 -m pip install seaborn
ls
pushd performance
python3 prof.py benchmarks/bench_ufunc_cupy.py -c

mkdir target
mv *.csv target/

# Run benchmarks for main branch
# Since GCP instance may change and use diff gen processsors/GPUs
# we just recompile and run to avoid false errors
python3 -m pip uninstall -y cupy

git clone https://github.com/cupy/cupy cupy-baseline
pushd cupy-baseline
if [[ "${PULL_REQUEST:-}" == "" ]]; then
    # For branches we compare against the latest release
    # TODO(ecastill) find a programatical way of doing this
    # sorting tags, or just checking the dates may mix the
    # stable & main branches
    git checkout tags/v11.0.0a2 -b v11.0.0a2
else
    git checkout main
fi
git submodule update --init
python3 -m pip install --user -v .
popd

python3 prof.py benchmarks/bench_ufunc_cupy.py -c

mkdir baseline
mv *.csv baseline/

# Compare with current branch
for bench in *.csv
do
    python3 regresion_detect.py baseline/${bench} target/${bench}
done

popd
