# Release Steps

* Bump `VERSION` in `setup.py`
* Build sdist: `docker run -u "$(id -u):$(id -g)" -v "${PWD}:${PWD}" -w "${PWD}" -e CUPY_UNIVERSAL_PKG_BUILD=1 python:3.10 python setup.py sdist`
* Upload sdist: `twine upload dist/cupy-wheel-*.tar.gz`
