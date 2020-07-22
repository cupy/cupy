#! /usr/bin/env sh
set -eux

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/
systemctl start docker.service

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/
cp -a . ${TEMP}/
cd ${TEMP}/

echo -n 3.6 | xargs -i -d ' ' -P $(nproc) sh -euxc '
PYTHON={}

docker build \
       --build-arg PYTHON=${PYTHON} \
       -t devel:${PYTHON} \
       .pfnci/docker/devel/
docker run --rm \
       --volume $(pwd):/cupy/ --workdir /cupy/ \
       devel:${PYTHON} \
       pip${PYTHON} wheel -e .
'

gsutil -q cp cupy-*.whl gs://tmp-asia-pfn-public-ci/cupy/wheel/${CI_COMMIT_ID}/cuda9.2/
echo ${CI_COMMIT_ID} | gsutil -q cp - gs://tmp-asia-pfn-public-ci/cupy/wheel/master
