#! /usr/bin/env sh
set -eux

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/
systemctl start docker.service

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/
cp -r . ${TEMP}/

echo -n 2.7 3.6 | xargs -i -d ' ' -P $(nproc) bash -euxc '
PYTHON={}

docker build \
       --build-arg PYTHON=${PYTHON} \
       -t devel:py${PYTHON//.} \
       .pfnci/docker/devel/
docker run --rm \
       --volume ${TEMP}/:/cupy/ --workdir /cupy/ \
       devel:py${PYTHON//.} \
       pip${PYTHON} wheel -e .
gsutil -q cp ${TEMP}/cupy-*-cp${PYTHON//.}-*.whl \
       gs://tmp-pfn-public-ci/cupy/wheel/${CI_COMMIT_ID}/cupy-cuda92-py${PYTHON//.}.whl
'

echo ${CI_COMMIT_ID} | gsutil -q cp - gs://tmp-pfn-public-ci/cupy/wheel/master
