# DGX Spark ComfyUI Forge

> **Status: highly experimental.** This repo is under active, hands-on
> tuning against one specific hardware/workflow combination (DGX Spark
> + MiniMax H3). Defaults, flags, and patches are still changing —
> check the [Changelog](#changelog) and [Known limitations](#known-limitations)
> before relying on it.

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
- ComfyUI runs through an isolated, persistent package venv's own
  Python interpreter, so both this project's dependency install loop
  *and* ComfyUI-Manager's live pip installs land in the same
  persistent location and survive container recreation.
- ComfyUI-Manager comes bundled via ComfyUI's own
  `manager_requirements.txt` (its version simply tracks whatever
  `COMFYUI_REF` ships) rather than a separate pinned git clone.
- Idempotent custom-node dependency installation, tracked with markers
  stored *inside* the isolated package venv rather than next to each
  node — so the markers and the venv can never drift out of sync, even
  across a manual venv reset (skips already-installed nodes on
  restart; auto-invalidates on image rebuild or venv recreation).
- `preflight-check.sh` — catches Bash syntax errors, corrupted patch
  files, invalid `docker-compose.yml`, and Dockerfile lint issues
  *before* a 30–90 minute build starts.
- Deliberately minimal runtime flags — every unified-memory tuning
  flag and environment variable was benchmarked against a real
  workload (MiniMax H3) rather than assumed; several commonly-
  recommended settings turned out to only cost speed without helping.
  See [design decisions](#design-decisions) below.

## Prerequisites

- NVIDIA DGX Spark (or other sm_121 / Blackwell GB10 hardware)
- Docker with the NVIDIA Container Toolkit configured (comes
  preinstalled on DGX OS)
- A models folder using ComfyUI's expected directory structure
  (`checkpoints/`, `vae/`, `loras/`, etc.) — this can be any folder on
  your system, it doesn't have to come from an existing ComfyUI
  installation, as long as it follows that structure. This repo mounts
  it, it doesn't manage downloading models.

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

### One-time Manager setup

ComfyUI-Manager ships with a conservative default that blocks
node/model installation from its web UI whenever ComfyUI is reachable
over the network (i.e. bound to `0.0.0.0`, as this setup always is) —
by design, since an installer reachable from your whole LAN is a
larger attack surface than one reachable only from `localhost`. To
allow installing custom nodes through the Manager UI, edit the
Manager's own config file (generated on first start, lives in your
persistent `/user` mount, **not** in this repo) and set:

```ini
[default]
network_mode = personal_cloud
```

Then restart ComfyUI (the Manager's own "Restart" button is enough, no
container recreate needed). If your DGX Spark is exposed beyond a
trusted home/LAN network, weigh this against your own threat model
first — `personal_cloud` mode relaxes several install-related
security checks network-wide, not just for you.

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
a persistent marker directory inside `/data`, not inside the container
itself.

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
<summary><strong>Why is ComfyUI-Manager installed via pip instead of a pinned git clone?</strong></summary>

Earlier versions of this project cloned `ComfyUI-Manager` directly
into `custom_nodes`, the traditional installation method most
ComfyUI Docker recipes still use. As of ComfyUI-Manager V4.0, upstream
changed the recommended installation method to a pip package
(`manager_requirements.txt`, shipped inside the ComfyUI repo itself)
plus a `--enable-manager` launch flag, and its docs now explicitly
warn against the old `git clone`-into-`custom_nodes` approach.
Our pinned `COMFYUI_REF` is recent enough to already ship
`manager_requirements.txt`, so we install through that instead — one
less thing to pin separately, and it avoids running two competing
Manager installs side by side.
</details>

<details>
<summary><strong>Why does ComfyUI launch through the venv's own Python interpreter?</strong></summary>

Earlier iterations launched ComfyUI via the base image's system
`python3`, with `PYTHONPATH` manually pointed at the isolated venv's
`site-packages` so custom-node dependencies were still importable.
This mostly worked for *running* ComfyUI, but broke ComfyUI-Manager's
own live "Install" button in two ways: pip/uv installs target
`sys.executable` by default, so Manager's installs landed in the
base image's system Python — which lives in the container's writable
layer and is **wiped on every `--force-recreate`**, silently undoing
any node installed through the UI. Worse, recent Debian/Ubuntu Python
installs are "externally managed" (PEP 668), and `uv`/`pip` flatly
refuse to install anything into the system interpreter outside of an
actual virtual environment - so Manager's live installs failed outright
with an `externally managed environment` error, not just a
persistence problem.

Launching through `$PKG_VENV/bin/python3` instead fixes both: since
the venv was created with `--system-site-packages`, it still sees the
base image's already-patched torch/xformers/etc., but pip/uv now see
an actual active venv as the install target - PEP 668 no longer
applies, and both this project's own dependency-install loop *and*
Manager's live installs land in the same persistent `/data/pkgvenv`,
surviving `--force-recreate`.
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
Python interpreter and target location - see the design decision above
on why ComfyUI is launched through this venv's interpreter directly.

The venv **must** be created with `--system-site-packages`. Without
it, the venv can't see the base image's already-correctly-patched,
sm_121-compatible torch/xformers — so installing any torch-dependent
custom-node requirement (e.g. `diffusers`, `accelerate`) silently pulls
a fresh, generic, sm_121-incompatible torch wheel from PyPI instead of
reusing the one already baked into the image.

Dependency-install markers are stored *inside* the venv
(`$PKG_VENV/.deps_installed_markers/`), not next to each node's
`requirements.txt`. This was a deliberate fix after a real incident:
manually deleting and recreating the venv (e.g. while troubleshooting
the `--system-site-packages` issue above) does **not** delete markers
that live elsewhere — a node's marker can end up claiming "already
installed" against a venv that never actually got that package,
silently breaking it. Keeping markers inside the venv means they can
never drift out of sync with it.
</details>

<details>
<summary><strong>Why so few runtime flags, compared to other DGX Spark ComfyUI setups?</strong></summary>

Earlier versions inherited a much larger flag/`CUDA_*`-env-var set from
another DGX Spark ComfyUI project. Benchmarking against a real workload
(MiniMax H3) showed none of it helped — sampling speed was unchanged
whether or not the extra flags were set, and several of them (notably
the `CUDA_*` block) actively cost performance without improving VRAM
stability, which came from the mmap patch and (at the time) disabling
DynamicVRAM instead — see the next design decision for how VRAM
stability is achieved now, with DynamicVRAM enabled. One recurring
lesson from benchmarking: short test runs (a few sampling steps) are
unreliable for comparing configs — one-off overhead (kernel warmup,
cuBLAS autotuning) dominates them. Only full-length runs were trusted
for the final call.

Current config: `--use-sage-attention --disable-mmap
--disable-pinned-memory` (DynamicVRAM/`comfy-aimdo` left **enabled** —
see the next design decision for why). Result: ~154 s/it average
sampling (10/10 steps), constant 94–96% GPU utilization, ~73–78 GB
VRAM peak, and a 46:50 → 32:56 minute drop in total end-to-end
generation time for a 15-second MiniMax H3 clip — matching a native
(non-Docker) reference install on the same hardware.
</details>

<details>
<summary><strong>Why is DynamicVRAM (comfy-aimdo) enabled, when an earlier version disabled it?</strong></summary>

An earlier iteration of this project disabled DynamicVRAM
(`--disable-dynamic-vram`) because it doesn't behave correctly on
unified memory: it's designed around a discrete VRAM/system-RAM
boundary that doesn't exist here, so it never triggers offloading and
VRAM grows unbounded during sampling. That fix worked, but had a real
cost: with DynamicVRAM disabled, ComfyUI's manual VRAM cleanup ("Free
model and node cache", "Unload Models", the `/free` API) stopped
working — `comfy-aimdo` is also responsible for the precise VRAM
tracking that cleanup depends on, and disabling it removes that
bookkeeping. The only reliable way to reclaim VRAM was a full
container recreate.

Digging into `comfy-aimdo`'s actual source (`comfy/model_management.py`,
`comfy_aimdo/host_buffer.py`) found the real lever: `aimdo` reserves a
pinned host-memory buffer per model, sized at up to 2× the model's
size from a `MAX_PINNED_MEMORY` ceiling that defaults to ~90% of total
system RAM — sensible on a discrete GPU, but on unified memory that
"system RAM" is the same physical pool VRAM lives in. `--disable-pinned-memory`
(an existing, official ComfyUI flag) skips that calculation entirely,
which starves this buffer down to zero without touching DynamicVRAM's
tracking. Result: DynamicVRAM stays enabled (cleanup works), VRAM peak
actually drops below the old `--disable-dynamic-vram` baseline
(~73–78 GB vs. ~88 GB), and sampling speed is unaffected. Verified
stable across a full run and repeated warm runs (no drift). A custom
patch to make the pinned-memory ceiling scale dynamically was
considered and dropped — an isolated test (halving the buffer's growth
cap) showed zero speed impact even at the old, much larger cap, so
there was no trade-off left for a patch to actually solve.
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
ignored, for certain models — and directly confirmed by our own
benchmark: `--bf16-vae` visibly changed the VAE's dtype in the logs,
but sampling speed was unaffected.
</details>

## Known limitations

- `ComfyUI-DGXSparkSafetensorsLoader` (a zero-copy loader that would
  fix the same double-memory issue our mmap patch addresses) loads
  without errors, but its compatibility with mixed-precision-quantized
  checkpoints (nvfp4 / int8-convrot, as used by e.g. MiniMax H3) is
  unverified — the loader reads quantization metadata expecting it on
  CPU, while quantized checkpoints may place it on GPU. Not needed
  here since the mmap patch already covers the memory issue for
  standard loading. A related project (`fastsafetensors`) explicitly
  documents that quantized models don't work with it, which is a
  strong signal the same limitation applies here.
- ComfyUI's built-in output gallery/asset browser loses track of
  existing files after a container restart — confirmed as an upstream
  ComfyUI bug (`Comfy-Org/ComfyUI#13061`, `#13078`), not specific to
  this setup; files are verified present on disk with correct
  permissions throughout. New outputs generated within an ongoing
  session do show up correctly; it's specifically the view after a
  restart that comes up empty. Several third-party nodes
  (`ComfyUI-Gallery`, `ComfyUI-AssetsPlus`) exist specifically to work
  around this. Not fixed here — either install one of those, or wait
  for the upstream fix to land in a future ComfyUI release.

## Changelog

Brief, dated summary of major changes — full reasoning behind each is
in the design decisions and known limitations above.

- **2026-08-26** — Solved the VRAM-cleanup / stable-sampling trade-off:
  `--disable-pinned-memory` with DynamicVRAM enabled gets both working
  manual cleanup and a lower, more stable VRAM peak than the previous
  `--disable-dynamic-vram` default, with no speed cost. New default
  config; see design decisions above.
- **2026-08-25** — Investigated `comfy-aimdo`/DynamicVRAM's VRAM
  usage vs. cleanup trade-off in depth. A tuned `--vram-headroom`
  value lowered VRAM usage but didn't generalize across workflows/
  resolutions, so it wasn't adopted (superseded by the 08-26 fix
  above).
- **2026-08-23** — Major performance pass: pruned an inherited
  `CUDA_*`/flag block that was costing speed without helping VRAM
  stability. Total end-to-end generation time for a 15s MiniMax H3
  clip dropped ~30% (46:50 → 32:56 min).
- **2026-08-22** — Migrated ComfyUI-Manager to the official pip-based
  install method (`manager_requirements.txt` + `--enable-manager`);
  ComfyUI now launches through its own venv's Python interpreter so
  Manager's live installs persist correctly. Full rebuild verified
  after a system purge; public GitHub release.

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
