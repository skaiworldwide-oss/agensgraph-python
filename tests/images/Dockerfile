# Test image: an official AgensGraph image plus pgvector, nothing else.
# Built wherever it is used -- in every CI run, with the build cache -- and never published.
# Always built for the machine's own architecture: every BASE is multi-arch, so FROM picks the native
# variant and pgvector compiles natively. Never cross-build with --platform; that needs QEMU.
# A cold build takes about 35 s, nearly all of it installing the toolchain; pgvector itself compiles
# in about 2 s.
#
#   docker build --build-arg BASE=skaiworldwide/agensgraph:v2.16.0-bookworm \
#     --build-arg AGVERSION=2.16.0 -t agensgraph-test:2.16.0 tests/images
#
# BASE per release (bookworm, multi-arch). The superuser is `postgres` up to 2.17 and `agens` on 2.18.
#   2.14.1 2.15.0 2.16.0 2.17.0 2.18.4.0 2.18.6.1   skaiworldwide/agensgraph:v<release>-bookworm
# PGVECTOR: 0.8.7 builds for PostgreSQL 14 through 18.
ARG BASE
ARG PGVECTOR=0.8.7

FROM ${BASE} AS build
ARG PGVECTOR
RUN set -eux; apt-get update; \
    apt-get install -y --no-install-recommends build-essential ca-certificates wget; \
    wget -qO /tmp/pgvector.tar.gz "https://github.com/pgvector/pgvector/archive/refs/tags/v${PGVECTOR}.tar.gz"; \
    mkdir /tmp/pgvector; tar -xzf /tmp/pgvector.tar.gz -C /tmp/pgvector --strip-components=1; \
    cd /tmp/pgvector; \
    # OPTFLAGS="": no -march=native, so the image runs on any CPU of its architecture.
    make -j"$(nproc)" OPTFLAGS="" PG_CONFIG=/usr/local/bin/pg_config; \
    make install OPTFLAGS="" PG_CONFIG=/usr/local/bin/pg_config DESTDIR=/out

FROM ${BASE}
ARG AGVERSION
ARG PGVECTOR
COPY --from=build /out/usr/local/ /usr/local/
COPY initdb/ /docker-entrypoint-initdb.d/
COPY healthcheck.sh /usr/local/bin/agensgraph-test-healthcheck
ENV AGENSGRAPH_TEST_EXPECT_VERSION=${AGVERSION}
HEALTHCHECK --interval=2s --timeout=5s --retries=30 CMD ["agensgraph-test-healthcheck"]
LABEL org.opencontainers.image.description="AgensGraph ${AGVERSION} + pgvector ${PGVECTOR}, for tests only"
