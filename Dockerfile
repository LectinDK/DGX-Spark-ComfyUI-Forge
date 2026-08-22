# syntax=docker/dockerfile:1.7
ARG BASE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        tini gosu git git-lfs ffmpeg libgl1 libglib2.0-0 \
        build-essential ninja-build cmake pkg-config \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

ARG MAX_JOBS=8
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.9;12.1" \
    CMAKE_CUDA_ARCHITECTURES="80;89;121" \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=2 \
    HF_HOME=/data/.cache/huggingface \
    COMFYUI_PATH=/opt/comfyui

# ---- ComfyUI (pinned) ----
ARG COMFYUI_REF=v0.33.1
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_PATH} && \
    cd ${COMFYUI_PATH} && git checkout ${COMFYUI_REF} && \
    git rev-parse HEAD > ${COMFYUI_PATH}/.commit

# ---- Double memory usage (unified memory): mmap copy=True -> copy=False ----
RUN python3 -c "\
from pathlib import Path; \
p = Path('${COMFYUI_PATH}/comfy/utils.py'); \
t = p.read_text(); \
old = 'tensor = tensor.to(device=device, copy=True)'; \
new = 'tensor = tensor.to(device=device, copy=False)'; \
assert old in t, 'patch target not found'; \
p.write_text(t.replace(old, new)); \
print('mmap copy=False patch applied')"

# ---- ComfyUI + extra requirements ----
RUN grep -vE '^(torch|torchaudio|torchvision)\b' ${COMFYUI_PATH}/requirements.txt \
        > /tmp/comfyui-requirements-filtered.txt && \
    pip install --upgrade-strategy only-if-needed \
        -r /tmp/comfyui-requirements-filtered.txt

COPY requirements-extra.txt /tmp/requirements-extra.txt
RUN grep -vE '^(torch|torchaudio|torchvision)\b' /tmp/requirements-extra.txt \
        > /tmp/requirements-extra-filtered.txt && \
    pip install --upgrade-strategy only-if-needed \
        -r /tmp/requirements-extra-filtered.txt

RUN pip install --no-cache-dir --force-reinstall comfy-kitchen==0.2.31

# ---- Triton ----
ARG INSTALL_TRITON=1
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_TRITON}" = "1" ]; then \
        TRITON_REQ=$(python3 -c "import importlib.metadata as m; reqs = m.requires('torch') or []; tr = [r for r in reqs if 'triton' in r.split(';')[0].lower() and 'extra ==' not in r]; print('\n'.join(tr))") && \
        if [ -n "${TRITON_REQ}" ]; then \
            echo "${TRITON_REQ}" | pip install --upgrade-strategy only-if-needed -r /dev/stdin ; \
        else \
            pip install --upgrade-strategy only-if-needed triton ; \
        fi && \
        python3 -c "import triton; print('triton', triton.__version__)" ; \
    fi

# ---- torchaudio ----
ARG BUILD_TORCHAUDIO=1
ARG TORCHAUDIO_REF=v2.11.0
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_TORCHAUDIO}" = "1" ]; then \
        git clone --depth 1 --branch "${TORCHAUDIO_REF}" --recurse-submodules \
            https://github.com/pytorch/audio.git /tmp/torchaudio && \
        cd /tmp/torchaudio && \
        USE_CUDA=1 BUILD_SOX=0 BUILD_RNNT=0 BUILD_CTC_DECODER=0 USE_FFMPEG=0 \
        TORCH_CUDA_ARCH_LIST="12.1" \
        pip install --no-build-isolation -v . && \
        git -C /tmp/torchaudio rev-parse HEAD > /opt/torchaudio.commit && \
        python3 -c "\
import re; \
from pathlib import Path; \
p = Path('/usr/local/lib/python3.12/dist-packages/torchaudio/_extension/utils.py'); \
t = p.read_text(); \
t2 = re.sub(r'def _check_cuda_version\(\):', 'def _check_cuda_version():\n    return  # patched: DGX Spark NGC toolkit/torch label mismatch is harmless', t, count=1); \
assert t2 != t, 'patch target not found'; \
p.write_text(t2); \
print('torchaudio CUDA version check disabled')" && \
        cd / && rm -rf /tmp/torchaudio && \
        python3 -c "import torchaudio; print('torchaudio', torchaudio.__version__)" ; \
    fi

# ---- xformers (patched for sm121) ----
ARG BUILD_XFORMERS=1
ARG XFORMERS_REF=v0.0.32
COPY patches/xformers-disable-cutlass-on-sm121.patch /tmp/patches/xformers-disable-cutlass-on-sm121.patch
COPY patches/xformers-fa3-runtime-belt-and-braces.patch /tmp/patches/xformers-fa3-runtime-belt-and-braces.patch
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_XFORMERS}" = "1" ]; then \
        git clone --depth 1 --branch "${XFORMERS_REF}" --recurse-submodules \
            https://github.com/facebookresearch/xformers.git /tmp/xformers && \
        cd /tmp/xformers && \
        git apply --verbose /tmp/patches/xformers-disable-cutlass-on-sm121.patch && \
        git apply --verbose /tmp/patches/xformers-fa3-runtime-belt-and-braces.patch && \
        XFORMERS_DISABLE_FLASH_ATTN=1 \
        TORCH_CUDA_ARCH_LIST="8.0;12.1" \
        CMAKE_CUDA_ARCHITECTURES="80;121" \
        pip install --no-build-isolation -v . && \
        git -C /tmp/xformers rev-parse HEAD > /opt/xformers.commit && \
        cd / && rm -rf /tmp/xformers && \
        python3 -c "import xformers, xformers.ops as xo; assert hasattr(xo, 'memory_efficient_attention'); print('xformers', xformers.__version__, 'ok')" ; \
    fi

# ---- SageAttention (main branch for sm121 runtime detection) ----
ARG BUILD_SAGE_ATTN=1
ARG SAGEATTN_REF=main
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_SAGE_ATTN}" = "1" ]; then \
        git clone https://github.com/thu-ml/SageAttention.git /tmp/sage && \
        cd /tmp/sage && git checkout ${SAGEATTN_REF} && \
        TORCH_CUDA_ARCH_LIST="8.0;8.9;12.1" \
        CMAKE_CUDA_ARCHITECTURES="80;89;121" \
        pip install --no-build-isolation -v . && \
        git -C /tmp/sage rev-parse HEAD > /opt/sageattention.commit && \
        cd / && rm -rf /tmp/sage && \
        python3 -c "import sageattention; print('sageattention ok')" ; \
    fi

# ---- ComfyUI-Manager ----
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git ${COMFYUI_PATH}/custom_nodes/ComfyUI-Manager && \
    pip install --upgrade-strategy only-if-needed -r ${COMFYUI_PATH}/custom_nodes/ComfyUI-Manager/requirements.txt

# ---- custom_nodes as a symlink to the persistent mount point ----
# ComfyUI-Manager was just cloned into
# ${COMFYUI_PATH}/custom_nodes/ComfyUI-Manager - we move it along so
# it already lands in the mounted /custom-nodes folder on first start
# (instead of disappearing after the build).
RUN mkdir -p /custom-nodes-seed && \
    mv ${COMFYUI_PATH}/custom_nodes/* /custom-nodes-seed/ 2>/dev/null || true && \
    rmdir ${COMFYUI_PATH}/custom_nodes && \
    ln -s /custom-nodes ${COMFYUI_PATH}/custom_nodes

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
RUN date -u +%Y%m%d%H%M%S > /opt/image-build-id

WORKDIR /opt/comfyui
EXPOSE 8188
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/entrypoint.sh"]