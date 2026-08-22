#!/usr/bin/env bash
set -e
cd ~/AI/docker-volumes/DGX-Spark-ComfyUI-Forge
./preflight-check.sh || exit 1
docker compose build dgx-spark-comfyui-forge
docker compose up -d --force-recreate dgx-spark-comfyui-forge
docker compose logs -f --timestamps