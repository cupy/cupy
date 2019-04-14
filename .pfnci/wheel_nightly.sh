#! /usr/bin/env sh
set -eux

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/
systemctl start docker.service
docker build -t devel .pfnci/docker/devel/

TEMP27=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP27}/
cp -r . ${TEMP27}/
docker run --rm \
       --volume ${TEMP27}/:/cupy/ --workdir /cupy/ \
       devel \
       python2.7 setup.py bdist_wheel &
PID27=$!

TEMP36=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP36}/
cp -r . ${TEMP36}/
docker run --rm \
       --volume ${TEMP36}/:/cupy/ --workdir /cupy/ \
       devel \
       python3.6 setup.py bdist_wheel &
PID36=$!

wait ${PID27} ${PID36}

gsutil -q cp ${TEMP27}/dist/cupy-*.whl gs://tmp-pfn-public-ci/cupy/wheel/${CI_COMMIT_ID}/cupy-cuda92-py27.whl
gsutil -q cp ${TEMP36}/dist/cupy-*.whl gs://tmp-pfn-public-ci/cupy/wheel/${CI_COMMIT_ID}/cupy-cuda92-py36.whl
echo ${CI_COMMIT_ID} | gsutil -q cp - gs://tmp-pfn-public-ci/cupy/wheel/master
