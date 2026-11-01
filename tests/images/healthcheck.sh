#!/bin/sh
# Healthy only when the server answers and reports the AgensGraph version the image was built for,
# so a mislabelled base image fails the job instead of testing the wrong line.
set -eu
user="${POSTGRES_USER:-$(id -un 2>/dev/null || echo postgres)}"
for candidate in "$user" postgres agens; do
    if got=$(psql -h 127.0.0.1 -U "$candidate" -d postgres -tAc "SHOW agversion" 2>/dev/null); then
        [ -z "${AGENSGRAPH_TEST_EXPECT_VERSION:-}" ] && exit 0
        [ "$got" = "$AGENSGRAPH_TEST_EXPECT_VERSION" ] && exit 0
        echo "agversion is $got, expected $AGENSGRAPH_TEST_EXPECT_VERSION" >&2
        exit 1
    fi
done
exit 1
