#!/usr/bin/env bash
set -uo pipefail
FAILED=0

echo "== [1/4] entrypoint.sh Syntax =="
if bash -n entrypoint.sh; then
    echo "  OK"
else
    echo "  FEHLER: entrypoint.sh hat einen Syntaxfehler"
    FAILED=1
fi

echo "== [2/4] Patch-Dateien Integrität =="
if sha256sum -c patches.sha256 --quiet; then
    echo "  OK"
else
    echo "  FEHLER: mindestens eine Patch-Datei weicht vom bekannten Stand ab (Copy-Paste-Fehler?)"
    FAILED=1
fi

echo "== [3/4] docker-compose.yml + .env =="
if docker compose config --quiet; then
    echo "  OK"
else
    echo "  FEHLER: docker-compose.yml/.env ungültig"
    FAILED=1
fi

echo "== [4/4] Dockerfile Lint =="
if docker build --check . ; then
    echo "  OK"
else
    echo "  FEHLER: Dockerfile-Probleme gefunden"
    FAILED=1
fi

echo ""
if [ "$FAILED" -eq 1 ]; then
    echo "❌ Vorab-Prüfung fehlgeschlagen - Build wird NICHT gestartet."
    exit 1
else
    echo "✅ Alle Prüfungen bestanden."
fi
