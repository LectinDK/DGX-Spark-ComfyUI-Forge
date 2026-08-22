# DGX Spark ComfyUI Forge

A Docker Compose setup for running [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
on the **NVIDIA DGX Spark** (GB10 / Grace-Blackwell, compute capability
sm_121), built on an NGC PyTorch base image with a patched xformers,
a self-compiled SageAttention, and torchaudio built from source.

## Why this exists

GB10 (sm_121) is very new hardware. At the time this project was built,
several common ComfyUI Docker recipes for it had real, sometimes silent
problems on this specific chip:

- Unpinned `ComfyUI` versions repeatedly hit an unstable `comfy_kitchen`
  integration on `master`.
- Compiling CUDA extensions with a single, isolated `-gencode` flag for
  sm_121 (instead of a full architecture *family* list) has been reported
  to cause **silent output corruption** — no crash, no error, just wrong
  results — in at least one other team's build.
- SageAttention's compiled kernels can be correct while its *runtime*
  architecture detection still doesn't recognize `sm_121` as a known
  string, silently falling back to slow PyTorch attention.
- The DGX Spark's unified memory architecture causes ComfyUI to load
  every model **twice** into memory unless a specific `mmap` behavior is
  patched.

This repo's Dockerfile and `entrypoint.sh` exist to work around these
issues in a documented, reproducible way — see
[`NOTICE.md`](./NOTICE.md) for attribution and
[design decisions](#design-decisions) below for the reasoning behind
each workaround.

## Features

- NGC PyTorch base image (`nvcr.io/nvidia/pytorch`) instead of a bare
  CUDA image, for verified-correct SDPA kernels on Blackwell.
- xformers built with two patches (disable CUTLASS on sm_121, disable
  Flash-Attention-3) — required because PyTorch's built-in CUTLASS
  kernels don't support sm_121.
- SageAttention compiled against the full architecture family
  (`8.0;8.9;12.1`), avoiding the silent-corruption risk of a bare
  `12.1`.
- torchaudio built from source, with NGC's non-standard CUDA version
  labeling patched around.
- mmap `copy=False` patch — fixes double memory usage on unified memory.
- Non-root execution via `gosu`, with automatic ownership repair on
  first start.
- Granular volume mounts (`/models`, `/output`, `/input`, `/user`,
  `/custom-nodes`, `/data`) instead of one shared workspace.
- Idempotent custom-node dependency installation (skips already-
  installed nodes on restart; auto-invalidates on image rebuild).
- `preflight-check.sh` — catches Bash syntax errors, corrupted patch
  files, invalid `docker-compose.yml`, and Dockerfile lint issues
  *before* a 30–90 minute build starts.

## Prerequisites

- NVIDIA DGX Spark (or other sm_121 / Blackwell GB10 hardware)
- Docker with the NVIDIA Container Toolkit configured (comes
  preinstalled on DGX OS)
- An existing ComfyUI models folder (`checkpoints/`, `vae/`, `loras/`,
  etc.) — this repo mounts your models, it doesn't manage downloading
  them

## Quick Start

```bash
git clone https://github.com/LectinDK/DGX-Spark-ComfyUI-Forge.git
cd DGX-Spark-ComfyUI-Forge

cp .env.example .env
nano .env   # set COMFYUI_HOST_PATH and FORGE_DATA_PATH to your paths

sha256sum patches/*.patch > patches.sha256   # baseline for integrity check

./preflight-check.sh   # validates config before building
./safe-rebuild.sh      # build + start (first build: ~30-90 minutes)
```

ComfyUI will be reachable at `http://<host>:8190` (or whatever port you
set in `.env`).

## Configuration

All settings live in `.env` — see [`.env.example`](./.env.example) for
the full list with inline comments. The two you must set:

| Variable | Purpose |
|---|---|
| `COMFYUI_HOST_PATH` | Path to your existing models folder (mounted read-write to `/models`) |
| `FORGE_DATA_PATH` | Persistent data folder for this setup (custom nodes, outputs, user settings) — **must be a different path than the cloned repo folder** |

Everything else (base image, pinned refs, build toggles, runtime flags)
has a sensible default and rarely needs changing.

## Day-to-day commands

| You changed... | Run this |
|---|---|
| `Dockerfile`, `entrypoint.sh`, `requirements-extra.txt`, `patches/*` | `./safe-rebuild.sh` (build + recreate) |
| `.env`, `docker-compose.yml` (env/volumes/ports) | `docker compose up -d --force-recreate` |

`docker compose restart` keeps the writable container layer (including
anything installed live via ComfyUI-Manager) but does **not** re-read
`.env`/`docker-compose.yml` changes. `--force-recreate` always builds a
fresh container — that's why custom-node dependencies are tracked with
a persistent `.deps_installed` marker in `/custom-nodes`, not inside
the container itself.

## Design decisions

<details>
<summary><strong>Why is SageAttention pinned to <code>main</code> instead of a tagged release?</strong></summary>

The tagged `v2.2.0` release compiles correctly for sm_121 when given
the right architecture flags, but its **runtime** architecture
detection (`get_cuda_arch_versions()` in `sageattention/core.py`)
doesn't recognize the string `"sm121"` — it falls back to slow PyTorch
attention with a logged warning, even though the compiled kernel is
fine. The `main` branch's detection code is newer and handles it
correctly. Because Docker caches `git clone --branch main` like any
other layer, the actual installed commit is *not* "whatever is newest
right now" — it's whatever was current the last time this layer was
rebuilt. Check `/opt/sageattention.commit` inside a running container
to see exactly which commit is in use.
</details>

<details>
<summary><strong>Why granular mounts instead of one shared /workspace?</strong></summary>

The original guide this project drew inspiration from was written for
a Kubernetes/PVC context, where one shared volume per pod is the path
of least resistance. In a single-host Docker Compose setup, splitting
mounts (`/models`, `/output`, `/input`, `/user`, `/custom-nodes`,
`/data`) makes ownership-repair operations smaller and more targeted,
and keeps large model files untouched by unrelated `chown` operations.
</details>

<details>
<summary><strong>Why a custom Python venv for custom-node dependencies?</strong></summary>

Two approaches were tried and rejected first:
- `pip install --user` failed silently because the NGC image has two
  separate Python installations (`/usr/local/bin/pip` vs.
  `/usr/bin/python3`) with different `site-packages` — packages landed
  somewhere ComfyUI's actual interpreter never looked.
- `pip install --target=...` caused multi-minute-plus hangs from pip's
  dependency resolver backtracking against NGC's enormous, tightly
  pinned package set.

An isolated venv (`/data/pkgvenv`) sidesteps both: it's a small,
mostly-empty namespace with almost nothing to conflict with, so
resolution is fast and correct, and both `pip install` calls (our
startup loop and ComfyUI-Manager's live installs) use the exact same
Python interpreter and target location.
</details>

<details>
<summary><strong>Why no --force-fp16 / --bf16-* flags?</strong></summary>

These flags are close to a no-op — or actively risky — for checkpoints
using mixed-precision quantization (nvfp4 / int8-convrot via
`comfy_kitchen`, as used by e.g. MiniMax H3). Those layers load through
a separate code path that reads embedded quantization metadata and
dispatches to dedicated native ops, bypassing the generic precision
flags entirely. This is corroborated by upstream ComfyUI issues
reporting the flags having no measurable effect, or being outright
ignored, for certain models.
</details>

## Known limitations

- `ComfyUI-DGXSparkSafetensorsLoader` (a zero-copy loader that would
  fix the same double-memory issue our mmap patch addresses) is
  currently incompatible with mixed-precision-quantized checkpoints —
  it loads quantization metadata onto the GPU, but the reading code
  requires it on CPU. Not needed here since the mmap patch already
  covers the memory issue for standard loading.
- ComfyUI-Manager may show `Revision: UNKNOWN` in its UI — cosmetic,
  caused by `custom_nodes` being a symlink into the persistent mount
  rather than a plain Git working directory. Does not affect
  functionality.

## Credits

See [`NOTICE.md`](./NOTICE.md) for third-party attribution — most
notably the xformers sm_121 patches and the overall build approach,
both originating from
[vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley).

## A note on how this was built

This project was developed collaboratively with Claude (Anthropic) —
debugging build issues, working out the sm_121-specific workarounds,
and drafting most of the code and docs. All design decisions and
testing on actual DGX Spark hardware are the author's own.

## License

GPL-3.0 — see [`LICENSE`](./LICENSE). The two patch files under
`patches/` remain under LGPL-3.0 (compatible per the
[FSF's own compatibility guidance](https://www.gnu.org/licenses/license-compatibility.html));
see `NOTICE.md`.
