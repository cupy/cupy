#! /usr/bin/env sh
set -eux

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/ -o size=75%
systemctl start docker.service
docker build -t devel .pfnci/docker/devel/

CCACHE=$(mktemp -d)
mount -t tmpfs tmpfs ${CCACHE}/ -o size=75%
gsutil -q cp gs://tmp-pfn-public-ci/cupy/ccache.tar - | tar -xf - -C ${CCACHE}/ || true
docker run --rm \
       --volume ${CCACHE}/:/root/.ccache/ \
       devel \
       ccache --max-size=1Gi --set-config=compiler_check=content

TEMP2=$(mktemp -d)
cp -r . ${TEMP2}/
docker run --rm \
       --volume ${TEMP2}/:/cupy/ --workdir /cupy/ \
       --volume ${CCACHE}/:/root/.ccache/ \
       devel \
       python2 setup.py bdist_wheel &
PID2=$!

TEMP3=$(mktemp -d)
cp -r . ${TEMP3}/
docker run --rm \
       --volume ${TEMP3}/:/cupy/ --workdir /cupy/ \
       --volume ${CCACHE}/:/root/.ccache/ \
       devel \
       python3 setup.py bdist_wheel &
PID3=$!

wait ${PID2} ${PID3}

tar -cf - -C ${CCACHE}/ -R . | gsutil -q cp - gs://tmp-pfn-public-ci/cupy/ccache.tar

gsutil -q cp ${TEMP2}/dist/* gs://tmp-pfn-public-ci/cupy/wheel/${CI_COMMIT_ID}/
gsutil -q cp ${TEMP3}/dist/* gs://tmp-pfn-public-ci/cupy/wheel/${CI_COMMIT_ID}/
echo ${CI_COMMIT_ID} | gsutil -q cp - gs://tmp-pfn-public-ci/cupy/wheel/master
