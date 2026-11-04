ARG BASE_IMAGE="rapidsai/base:26.10-cuda12-py3.11"
FROM ${BASE_IMAGE}

USER root

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install build-essential cuda-toolkit-12-9 ccache git curl

# Fail the build if a CA bundle still trusts the compromised eMudhra root
# (https://discuss.python.org/t/109576). Checks the system store, conda's
# bundle and both Python certifi copies. Uses openssl to read subjects from
# the raw PEM files, which have no label comments that text-search could use.
# Covered by the rapidsai/base:26.10 hotfix (rapidsai/docker#923).
RUN for f in /etc/ssl/certs/ca-certificates.crt \
             /opt/conda/ssl/cert.pem \
             "$(python -m certifi)" \
             "$(python -c 'import pip._vendor.certifi as c; print(c.where())')"; do \
        test -s "$f" || { echo "missing CA bundle $f"; exit 1; }; \
        if openssl crl2pkcs7 -nocrl -certfile "$f" | openssl pkcs7 -print_certs -noout \
                | grep -q 'CN *= *emSign ECC Root CA - G3'; then \
            echo "compromised eMudhra CA in $f"; exit 1; \
        fi; \
    done

ENV CUDA_PATH "/usr/local/cuda"
ENV PATH "/usr/lib/ccache:${PATH}"
