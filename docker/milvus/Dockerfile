# Recreate the official v2.6.24 runtime on a separately verified Ubuntu layer.
# The local upstream Ubuntu layer contained zero-byte shared libraries. Keeping
# only the upstream Milvus/Tini artifacts avoids carrying that damaged OS layer.
# Runtime dependencies and environment follow milvus-io/milvus v2.6.24:
# build/docker/milvus/ubuntu22.04/Dockerfile.
FROM milvusdb/milvus:v2.6.24@sha256:c55810ca95c97437cc003f913b835524600fb95a14ce04955a92517c3ccab73f AS upstream
FROM ubuntu:22.04@sha256:b8b6ee6aa931ecd9d0d952abc34dc0e5f7c6a30c6bb71b079fe399fde0329c02

# apt verifies signed Ubuntu archive metadata and each downloaded package hash.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl libaio-dev libgomp1 libopenblas-dev tzdata && \
    ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    echo "Etc/UTC" > /etc/timezone && \
    dpkg-query -W > /cloudcare-runtime-packages.txt && \
    rm -rf /var/lib/apt/lists/*

COPY --from=upstream /tini /tini
COPY --from=upstream /milvus /milvus

ENV PATH=/milvus/bin:$PATH \
    LD_LIBRARY_PATH=/milvus/lib:/usr/lib \
    LD_PRELOAD=/milvus/lib/libjemalloc.so \
    MALLOC_CONF=background_thread:true \
    OPENSSL_MODULES=/milvus/lib/ossl-modules

# Fail the build on the exact previously broken libraries and health executable.
RUN test -s /lib/x86_64-linux-gnu/libz.so.1 && \
    test -s /lib/x86_64-linux-gnu/libselinux.so.1 && \
    test -s /milvus/bin/milvus && \
    curl --version && /tini --version && \
    test -z "$(find /milvus/lib -type f -name '*.so*' -size 0)"

WORKDIR /milvus
ENTRYPOINT ["/tini", "--"]
