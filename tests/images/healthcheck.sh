#!/bin/sh
# Healthy only when the server answers and reports the AgensGraph version the image was built for,
# so a mislabelled base image fails the job instead of testing the wrong line.
set -eu
expected="${AGENSGRAPH_TEST_EXPECT_VERSION:-}"
# Two releases report their version in another form: 2.14.1's image reports 2.14.0, and 2.16.0's
# reports the line alone.
case "$expected" in
    2.14.1) expected=2.14.0 ;;
    2.16.0) expected=2.16 ;;
esac
# The image's superuser: POSTGRES_USER where it is set, postgres up to 2.17, agens on 2.18.
for candidate in ${POSTGRES_USER:-} postgres agens; do
    if got=$(psql -h 127.0.0.1 -U "$candidate" -d postgres -tAc "SHOW agversion" 2>/dev/null); then
        [ -z "$expected" ] && exit 0
        [ "$got" = "$expected" ] && exit 0
        echo "agversion is $got, expected $expected" >&2
        exit 1
    fi
done
exit 1
