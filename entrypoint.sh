#!/usr/bin/env bash
set -euo pipefail

APP=/opt/comfyui
PY="$(command -v python3 || command -v python)"
PKG_VENV=/data/pkgvenv

# On the very first start: copy ComfyUI-Manager (baked into the image)
# into the still-empty, persistent /custom-nodes mount.
if [ -d /custom-nodes-seed ] && [ -z "$(ls -A /custom-nodes 2>/dev/null)" ]; then
    echo "[entrypoint] Seeding /custom-nodes with ComfyUI's bundled default files..."
    cp -a /custom-nodes-seed/. /custom-nodes/
fi

# All mounted folders may have just been freshly created by Docker as
# root - fix that here before we switch to user (1000).
# Deliberately NOT recursive for models (can be huge).
# chown 1000:1000 /output /input /user /custom-nodes /data 2>/dev/null || true
chown -R 1000:1000 /output /input /user /custom-nodes /data 2>/dev/null || true

# Clean up leftover .deps_installed marker files from before dependency
# markers moved into $PKG_VENV/.deps_installed_markers/ (see Idempotenz
# notes) - no longer read by anything, just harmless litter left behind
# by custom nodes installed under the old scheme. Cheap to check every
# boot; becomes a no-op once cleaned up (or on a fresh install, which
# never creates these files in the first place).
find /custom-nodes -maxdepth 2 -name ".deps_installed" -delete 2>/dev/null || true

# Seed ComfyUI-Manager's config.ini with use_uv=False, but only if the
# file doesn't exist yet (Manager creates it itself on first boot and
# fills in its own additional defaults around whatever's already there
# - it doesn't overwrite existing keys). Reason: uv's `pip list`/`pip
# show` don't correctly enumerate packages inherited via
# --system-site-packages (see astral-sh/uv#2500), which makes Manager
# log a confusing (but harmless) "PyTorch is not installed" during its
# own package checks. Plain pip respects --system-site-packages
# correctly, so this avoids the false report entirely.
MANAGER_CONFIG_DIR="/user/__manager"
MANAGER_CONFIG_FILE="$MANAGER_CONFIG_DIR/config.ini"
if [ ! -f "$MANAGER_CONFIG_FILE" ]; then
    echo "[entrypoint] Seeding ComfyUI-Manager config.ini with use_uv=False..."
    mkdir -p "$MANAGER_CONFIG_DIR"
    cat > "$MANAGER_CONFIG_FILE" <<EOF
[default]
use_uv = False
EOF
    chown -R 1000:1000 "$MANAGER_CONFIG_DIR"
fi

# On every new image build (new build ID), discard dependency-install
# markers so requirements are guaranteed to be reinstalled after a
# rebuild. Markers live inside $PKG_VENV/.deps_installed_markers (see
# below) rather than next to each node's requirements.txt, so that
# deleting/recreating the venv (e.g. a manual `rm -rf` for
# troubleshooting) automatically invalidates them too - a leftover
# marker from an old venv previously made the install loop skip
# reinstalling a node's deps into a freshly recreated venv, silently
# leaving it broken (fastsafetensors incident, 2026-08-22).
IMAGE_BUILD_ID="$(cat /opt/image-build-id 2>/dev/null || echo unknown)"
STAMP_FILE=/data/.last-image-build-id
if [ ! -f "$STAMP_FILE" ] || [ "$(cat "$STAMP_FILE")" != "$IMAGE_BUILD_ID" ]; then
    echo "[entrypoint] New image build detected ($IMAGE_BUILD_ID) - clearing dependency-install markers"
    rm -rf "$PKG_VENV/.deps_installed_markers" 2>/dev/null || true
    echo "$IMAGE_BUILD_ID" > "$STAMP_FILE"
fi

if [ ! -f "$APP/extra_model_paths.yaml" ]; then
    cat > "$APP/extra_model_paths.yaml" <<EOF
comfyui:
    base_path: /models
    audio_encoders: audio_encoders
    background_removal: background_removal
    checkpoints: checkpoints
    clip: clip
    clip_vision: clip_vision
    configs: configs
    controlnet: controlnet
    detection: detection
    diffusers: diffusers
    diffusion_models: diffusion_models
    embeddings: embeddings
    frame_interpolation: frame_interpolation
    geometry_estimation: geometry_estimation
    gligen: gligen
    hypernetworks: hypernetworks
    latent_upscale_models: latent_upscale_models
    loras: loras
    model_patches: model_patches
    optical_flow: optical_flow
    photomaker: photomaker
    style_models: style_models
    text_encoders: text_encoders
    unet: unet
    upscale_models: upscale_models
    vae: vae
    vae_approx: vae_approx
EOF
fi

if [ ! -d "$PKG_VENV" ]; then
    echo "[entrypoint] Creating isolated package venv (with access to base image's torch/xformers/etc.)..."
    gosu 1000:1000 "$PY" -m venv --system-site-packages "$PKG_VENV"
fi

DEPS_MARKER_DIR="$PKG_VENV/.deps_installed_markers"
mkdir -p "$DEPS_MARKER_DIR"
chown 1000:1000 "$DEPS_MARKER_DIR"

for req in /custom-nodes/*/requirements.txt; do
    if [[ -f "$req" ]]; then
        node_name="$(basename "$(dirname "$req")")"
        marker="$DEPS_MARKER_DIR/$node_name"
        if [[ ! -f "$marker" ]]; then
            echo "[entrypoint] Installing deps from: $req"
            if gosu 1000:1000 "$PKG_VENV/bin/pip" install \
                --upgrade-strategy only-if-needed -r "$req"; then
                gosu 1000:1000 touch "$marker"
            else
                echo "[entrypoint] WARNING: pip install FAILED for $req - marker NOT set, will retry on next start"
            fi
        fi
    fi
done

echo "[entrypoint] ComfyUI commit: $(cat "$APP/.commit" 2>/dev/null || echo unknown)"
echo "[entrypoint] torch (via venv interpreter): $("$PKG_VENV/bin/python3" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability() if torch.cuda.is_available() else "n/a")')"

cd "$APP"
exec env HOME=/data gosu 1000:1000 "$PKG_VENV/bin/python3" main.py \
    --listen 0.0.0.0 \
    --port "${COMFYUI_PORT:-8188}" \
    --output-directory /output \
    --input-directory /input \
    --user-directory /user \
    --temp-directory /data/temp \
    --extra-model-paths-config "$APP/extra_model_paths.yaml" \
    ${COMFYUI_EXTRA_ARGS:-}