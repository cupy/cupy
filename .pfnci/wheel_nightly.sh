#! /usr/bin/env sh
set -eux

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/
systemctl start docker.service

echo -n 2.7 3.6 | xargs -i -d ' ' -P $(nproc) sh -euxc '
PYTHON={}

docker build \
       --build-arg PYTHON=${PYTHON} \
       -t devel:py${PYTHON//.} \
       .pfnci/docker/devel/

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/
cp -r . ${TEMP}/
docker run --rm \
       --volume ${TEMP}/:/cupy/ --workdir /cupy/ \
       devel:py${PYTHON//.} \
       python${PYTHON} setup.py bdist_wheel

gsutil -q cp ${TEMP}/dist/cupy-*.whl \
       gs://tmp-pfn-public-ci/cupy/wheel/${CI_COMMIT_ID}/cupy-cuda92-py${PYTHON//.}.whl
'

echo ${CI_COMMIT_ID} | gsutil -q cp - gs://tmp-pfn-public-ci/cupy/wheel/master
