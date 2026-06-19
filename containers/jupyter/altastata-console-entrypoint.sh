#!/bin/bash
# Auto-start altastata-grpc-server alongside Jupyter Lab when
# ENABLE_ALTASTATA_CONSOLE_UI=1. See mycloud/altastata-grpc/TLS_DESIGN.md.
set -e

if [ "${ENABLE_ALTASTATA_CONSOLE_UI:-1}" = "1" ]; then
    altastata-grpc-server > /tmp/altastata-grpc-server.log 2>&1 &
fi

exec "$@"
