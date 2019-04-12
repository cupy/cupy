#! /usr/bin/env sh
set -eux

docker build -t devel .pfnci/docker/devel/

TEMP2=$(mktemp -d)
cp -r . ${TEMP2}
docker run --rm \
       --volume ${TEMP2}:/cupy/ --workdir /cupy/ \
       devel \
       python2 setup.py bdist_wheel &
PID2=$!

TEMP3=$(mktemp -d)
cp -r . ${TEMP3}
docker run --rm \
       --volume ${TEMP3}:/cupy/ --workdir /cupy/ \
       devel \
       python3 setup.py bdist_wheel &
PID3=$!

wait ${PID2} ${PID3}

if [ -n ${CI_COMMIT_ID:-} ]; then
    gsutil cp -q ${TEMP2}/dist/* gs://tmp-pfn-public-ci/cupy/nightly/${CI_COMMIT_ID}/
    gsutil cp -q ${TEMP3}/dist/* gs://tmp-pfn-public-ci/cupy/nightly/${CI_COMMIT_ID}/
    echo ${CI_COMMIT_ID} | gsutil cp -q - gs://tmp-pfn-public-ci/cupy/nightly/master
fi
