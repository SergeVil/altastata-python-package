#!/usr/bin/env bash
# Build the binary artifacts that ship inside the altastata wheel.
#
# Why this exists:
#   The altastata Python package ships two binary artifacts that this repo
#   does not store in git:
#     1. altastata-services-<ver>-uber.jar  (built from sibling
#        mycloud/altastata-services — the unified Micronaut app that hosts
#        gRPC + S3 + py4j under com.altastata.services.AltaStataServicesApplication)
#     2. altastata/lib/altastata-console-static/  (built from altastata-console/frontend)
#   Both are deliberately gitignored under altastata/lib/ so the repo stays
#   text-only. They are populated locally before `python -m build` so they
#   end up inside the wheel published to PyPI.
#
# Usage:
#   bash scripts/build-bundled-artifacts.sh
#
# Optional environment variables:
#   ALTASTATA_MYCLOUD_DIR   override path to the mycloud checkout
#                           (default: ../mycloud relative to this repo)
#   ALTASTATA_CONSOLE_DIR   override path to the altastata-console checkout
#                           (default: altastata-console/ in this repo)
#   SKIP_GRPC=1             skip the altastata-services Gradle build (use existing
#                           altastata-services-*-uber.jar already present in lib/)
#   SKIP_UI=1               skip the altastata-console npm build (use existing
#                           altastata/lib/altastata-console-static/ contents)
#
# What this does NOT do:
#   - It does not commit anything. The point is to populate altastata/lib/
#     locally so subsequent `pip install -e .` or `python -m build` picks
#     the artifacts up via setup.py's package_data globs.
#   - It does not touch py4j0.10.9.5.jar (the only jar this repo intentionally
#     tracks) or the Bouncy Castle jars that already live under altastata/lib/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB_DIR="$PKG_DIR/altastata/lib"
STATIC_DIR="$LIB_DIR/altastata-console-static"

MYCLOUD_DIR="${ALTASTATA_MYCLOUD_DIR:-$(cd "$PKG_DIR/../mycloud" 2>/dev/null && pwd || true)}"
CONSOLE_DIR="${ALTASTATA_CONSOLE_DIR:-$(cd "$PKG_DIR/altastata-console" 2>/dev/null && pwd || true)}"

mkdir -p "$LIB_DIR"

if [[ -z "${SKIP_GRPC:-}" ]]; then
    if [[ -z "$MYCLOUD_DIR" || ! -d "$MYCLOUD_DIR" ]]; then
        echo "ERROR: mycloud directory not found." >&2
        echo "       Set ALTASTATA_MYCLOUD_DIR or place mycloud next to this repo." >&2
        exit 1
    fi
    echo "==> Building altastata-services uber jar in $MYCLOUD_DIR"
    (cd "$MYCLOUD_DIR" && ./gradlew :altastata-services:shadowJar)

    UBER_JAR="$(ls -1 "$MYCLOUD_DIR"/altastata-services/build/libs/altastata-services-*-uber.jar 2>/dev/null | tail -1 || true)"
    if [[ -z "$UBER_JAR" || ! -f "$UBER_JAR" ]]; then
        echo "ERROR: no altastata-services-*-uber.jar produced under $MYCLOUD_DIR/altastata-services/build/libs/" >&2
        exit 1
    fi

    # Drop any previously-staged uber jars (both the new services name and the
    # legacy altastata-grpc name) so we never ship two side by side; the
    # launcher classpath would otherwise pick whichever sorts last.
    find "$LIB_DIR" -maxdepth 1 \( -name 'altastata-services-*-uber.jar' -o -name 'altastata-grpc-*-uber.jar' \) -print -delete
    cp "$UBER_JAR" "$LIB_DIR/"
    echo "    copied $(basename "$UBER_JAR") -> altastata/lib/"

    # Co-locate the BouncyCastle jars that altastata-services externalizes
    # (they are referenced from the uber jar manifest Class-Path) so JCE
    # signing keeps working when Python launches the gateway from altastata/lib.
    SERVICES_LIB_DIR="$MYCLOUD_DIR/altastata-services/build/libs/lib"
    if [[ -d "$SERVICES_LIB_DIR" ]]; then
        for bc_jar in "$SERVICES_LIB_DIR"/bcpkix-*.jar "$SERVICES_LIB_DIR"/bcprov-*.jar "$SERVICES_LIB_DIR"/bcutil-*.jar; do
            [[ -f "$bc_jar" ]] || continue
            cp "$bc_jar" "$LIB_DIR/"
            echo "    copied $(basename "$bc_jar") -> altastata/lib/"
        done
    fi
else
    echo "==> SKIP_GRPC=1, leaving altastata-services uber jar untouched"
fi

if [[ -z "${SKIP_UI:-}" ]]; then
    if [[ -z "$CONSOLE_DIR" || ! -d "$CONSOLE_DIR" ]]; then
        echo "ERROR: altastata-console directory not found." >&2
        echo "       Set ALTASTATA_CONSOLE_DIR or ensure altastata-console/ exists in this repo." >&2
        exit 1
    fi
    echo "==> Building altastata-console SPA in $CONSOLE_DIR/frontend"
    (cd "$CONSOLE_DIR/frontend" && npm install --no-audit --no-fund && npm run build)

    DIST_DIR="$CONSOLE_DIR/frontend/dist"
    if [[ ! -f "$DIST_DIR/index.html" ]]; then
        echo "ERROR: $DIST_DIR/index.html missing after npm run build" >&2
        exit 1
    fi

    rm -rf "$STATIC_DIR"
    mkdir -p "$STATIC_DIR"
    cp -R "$DIST_DIR"/. "$STATIC_DIR"/
    echo "    copied $DIST_DIR/* -> altastata/lib/altastata-console-static/"
else
    echo "==> SKIP_UI=1, leaving altastata-console-static untouched"
fi

echo
echo "==> altastata/lib/ contents:"
ls -1 "$LIB_DIR" | sed 's/^/    /'
if [[ -d "$STATIC_DIR" ]]; then
    echo "==> altastata/lib/altastata-console-static/ summary:"
    if [[ -f "$STATIC_DIR/VERSION" ]]; then
        echo "    VERSION = $(cat "$STATIC_DIR/VERSION")"
    fi
    echo "    files  = $(find "$STATIC_DIR" -type f | wc -l | tr -d ' ')"
    echo "    bytes  = $(du -sh "$STATIC_DIR" | awk '{print $1}')"
fi
