#!/usr/bin/env bash
set -uo pipefail
FAILED=0

echo "== [1/4] entrypoint.sh syntax =="
if bash -n entrypoint.sh; then
    echo "  OK"
else
    echo "  ERROR: entrypoint.sh has a syntax error"
    FAILED=1
fi

echo "== [2/4] Patch file integrity =="
if sha256sum -c patches.sha256 --quiet; then
    echo "  OK"
else
    echo "  ERROR: at least one patch file differs from the known-good state (copy/paste error?)"
    FAILED=1
fi

echo "== [3/4] docker-compose.yml + .env =="
if docker compose config --quiet; then
    echo "  OK"
else
    echo "  ERROR: docker-compose.yml/.env is invalid"
    FAILED=1
fi

echo "== [4/4] Dockerfile lint =="
if docker build --check . ; then
    echo "  OK"
else
    echo "  ERROR: Dockerfile issues found"
    FAILED=1
fi

echo ""
if [ "$FAILED" -eq 1 ]; then
    echo "❌ Preflight check failed - build will NOT start."
    exit 1
else
    echo "✅ All checks passed."
fi