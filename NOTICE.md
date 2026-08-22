# Third-Party Components

## patches/*.patch

The two patch files under `patches/` —
`xformers-disable-cutlass-on-sm121.patch` and
`xformers-fa3-runtime-belt-and-braces.patch` — are copied **unmodified**
from [vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley)
(`scripts/comfyui/patches/`), used here under **LGPL-3.0**
(compatible with this repository's GPL-3.0 license per the FSF's
own compatibility guidance — see
https://www.gnu.org/licenses/license-compatibility.html).

Neither file contains its own embedded copyright header; the origin
repository's default LGPL policy applies (see their
[LICENSE.md](https://github.com/vroomfondel/dgxarley/blob/main/LICENSE.md)).

## Build approach inspiration

The overall Docker build approach (NGC base image, xformers patches
for sm121, SageAttention architecture-family compilation flags) was
inspired by the same project's `COMFYUI_ARM64_SM121.md` guide.

All other files in this repository — Dockerfile, entrypoint.sh,
docker-compose.yml, preflight-check.sh, and associated scripts —
are original work.
