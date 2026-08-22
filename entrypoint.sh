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
# Launch through the venv's OWN interpreter (not the base system python3
# with a manual PYTHONPATH append). Because the venv was created with
# --system-site-packages, it still sees the base image's torch/xformers/
# etc. - but now ComfyUI-Manager's live pip-installs (which target
# sys.executable) correctly land in this persistent venv instead of the
# ephemeral system Python at /usr, and are no longer blocked by pip/uv's
# PEP 668 "externally managed environment" protection (which only
# applies outside of an active venv).
exec env HOME=/data gosu 1000:1000 "$PKG_VENV/bin/python3" main.py \
    --listen 0.0.0.0 \
    --port "${COMFYUI_PORT:-8188}" \
    --output-directory /output \
    --input-directory /input \
    --user-directory /user \
    --temp-directory /data/temp \
    --extra-model-paths-config "$APP/extra_model_paths.yaml" \
    ${COMFYUI_EXTRA_ARGS:-}