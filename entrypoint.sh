#!/usr/bin/env bash
set -euo pipefail

APP=/opt/comfyui
PY="$(command -v python3 || command -v python)"

# On the very first start: copy ComfyUI-Manager (baked into the image)
# into the still-empty, persistent /custom-nodes mount.
if [ -d /custom-nodes-seed ] && [ -z "$(ls -A /custom-nodes 2>/dev/null)" ]; then
    echo "[entrypoint] Seeding /custom-nodes with baked-in ComfyUI-Manager..."
    cp -a /custom-nodes-seed/. /custom-nodes/
fi

# All mounted folders may have just been freshly created by Docker as
# root - fix that here before we switch to user (1000).
# Deliberately NOT recursive for models (can be huge).
# chown 1000:1000 /output /input /user /custom-nodes /data 2>/dev/null || true
chown -R 1000:1000 /output /input /user /custom-nodes /data 2>/dev/null || true

# On every new image build (new build ID), discard all markers so
# requirements are guaranteed to be reinstalled after a rebuild.
IMAGE_BUILD_ID="$(cat /opt/image-build-id 2>/dev/null || echo unknown)"
STAMP_FILE=/data/.last-image-build-id
if [ ! -f "$STAMP_FILE" ] || [ "$(cat "$STAMP_FILE")" != "$IMAGE_BUILD_ID" ]; then
    echo "[entrypoint] New image build detected ($IMAGE_BUILD_ID) - clearing .deps_installed markers"
    find /custom-nodes -name ".deps_installed" -delete 2>/dev/null || true
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

PKG_VENV=/data/pkgvenv
if [ ! -d "$PKG_VENV" ]; then
    echo "[entrypoint] Creating isolated package venv..."
    gosu 1000:1000 "$PY" -m venv "$PKG_VENV"
fi

export PYTHONPATH="/data/python-packages${PYTHONPATH:+:$PYTHONPATH}"

for req in /custom-nodes/*/requirements.txt; do
    if [[ -f "$req" ]]; then
        marker="$(dirname "$req")/.deps_installed"
        if [[ ! -f "$marker" ]]; then
            echo "[entrypoint] Installing deps from: $req"
            gosu 1000:1000 "$PKG_VENV/bin/pip" install -q \
                --upgrade-strategy only-if-needed -r "$req" || true
            touch "$marker"
        fi
    fi
done

echo "[entrypoint] ComfyUI commit: $(cat "$APP/.commit" 2>/dev/null || echo unknown)"
PY="$(command -v python3 || command -v python)"
echo "[entrypoint] torch: $("$PY" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability() if torch.cuda.is_available() else "n/a")')"

cd "$APP"
PKG_VENV_SITE="$PKG_VENV/lib/python3.12/site-packages"
exec env HOME=/data PYTHONPATH="$PKG_VENV_SITE" gosu 1000:1000 "$PY" main.py \
    --listen 0.0.0.0 \
    --port "${COMFYUI_PORT:-8188}" \
    --output-directory /output \
    --input-directory /input \
    --user-directory /user \
    --temp-directory /data/temp \
    --extra-model-paths-config "$APP/extra_model_paths.yaml" \
    ${COMFYUI_EXTRA_ARGS:-}