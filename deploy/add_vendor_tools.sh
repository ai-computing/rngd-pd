#!/usr/bin/env bash
# Add the vendor CLI tools and runtime libraries the SDKs shell out to.
#
# Both stacks probe the machine before compiling: rebel-compiler asks the RBLN
# tools which NPU is present (without them it refuses with "Please specify
# `npu`"), and the Furiosa stack talks to furiosa-smi. The cluster's existing
# RBLN pods solve this by mounting the host's /usr/bin; an image has to carry
# the binaries instead.
#
# Device access still comes from the device plugins at run time, and the RBLN
# daemon socket (/var/run/rbln.sock) is mounted in, not baked.
set -euo pipefail

DEPS="libdrm2 libnuma1 libgomp1"

add() {                        # $1 image, rest: host paths to copy into /usr/bin
  local img=$1; shift
  local c
  c=$(buildah from "$img")
  buildah run "$c" -- sh -c \
    "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $DEPS && rm -rf /var/lib/apt/lists/*" >/dev/null
  for f in "$@"; do
    [ -e "$f" ] && buildah copy "$c" "$f" /usr/bin/ >/dev/null
  done
  buildah commit --rm "$c" "$img"
  echo "TOOLED $img"
}

add localhost/rngd-pd-decode:latest \
    /usr/bin/rbln-smi /usr/bin/rbln-stat /usr/bin/rbln-smdi /usr/bin/rblnvs

add localhost/rngd-pd-prefill:latest \
    /usr/bin/furiosa-smi /usr/local/bin/furiosa_rngd_drv_tools.sh

echo TOOLS_DONE
