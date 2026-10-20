#!/usr/bin/env bash
# Build the PD worker images with rootless buildah.
#
# Two images, because the two roles need different vendor stacks and there is no
# reason to ship 12 GB to a pod that only needs half of it:
#
#   rngd-pd-prefill  Furiosa: the venv holding furiosa-torch/furiosa-llm, plus
#                    libfuriosa_smi from the host packages.
#   rngd-pd-decode   Rebellions: the user site-packages holding rebel-compiler,
#                    tvm/librbln, torch and transformers, plus the librbln
#                    runtime libraries.
#
# Both take python3.10 from Ubuntu 22.04 so the cp310 extension modules copied
# off this host keep working. Neither image contains model weights — those stay
# on the host and are mounted, since they are 16 GB and licensed separately.
#
# buildah copies straight from host paths, so there is no build context to stage
# (a docker build would have to copy 12 GB into a context directory first).
set -euo pipefail

REPO=${REPO:-$HOME/workspace/01_pd_disag/rngd_llmd}
VENV=${VENV:-$HOME/venvs/furiosa}
USERSITE=${USERSITE:-$HOME/.local/lib/python3.10/site-packages}
TAG=${TAG:-latest}
APT="python3.10 python3.10-distutils libpython3.10 ca-certificates"

base() {                       # $1 = image name -> prints working container id
  local c
  c=$(buildah from docker.io/library/ubuntu:22.04)
  buildah run "$c" -- sh -c \
    "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $APT && rm -rf /var/lib/apt/lists/*" >/dev/null
  buildah run "$c" -- ln -sf /usr/bin/python3.10 /usr/bin/python3
  echo "$c"
}

echo "== decode image (Rebellions ATOM) =="
c=$(base)
buildah copy "$c" "$USERSITE" /opt/py >/dev/null
for so in /usr/lib/librbln-thunk.so* /usr/lib/librbln-ml.so* /usr/lib/librbln-ccl.so*; do
  buildah copy "$c" "$so" /usr/lib/ >/dev/null
done
buildah copy "$c" "$REPO" /app >/dev/null
buildah config \
  --env PYTHONPATH=/opt/py \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env VLLM_PLUGINS= \
  --env HF_HOME=/hf \
  --env MODEL=meta-llama/Llama-3.1-8B-Instruct \
  --env PD_SHM_ROOT=/dev/shm/rngd_kv_cluster \
  --workingdir /app \
  --entrypoint '["python3.10","/app/serving/pd_cluster.py"]' \
  --cmd '["--role","decode","--backend","atom","--port","8000"]' "$c"
buildah commit --rm "$c" "rngd-pd-decode:$TAG"

echo "== prefill image (Furiosa RNGD) =="
c=$(base)
buildah copy "$c" "$VENV" /venv >/dev/null
buildah copy "$c" "$USERSITE" /opt/py >/dev/null
for so in /usr/lib/x86_64-linux-gnu/libfuriosa_smi.so*; do
  buildah copy "$c" "$so" /usr/lib/x86_64-linux-gnu/ >/dev/null
done
buildah copy "$c" "$REPO" /app >/dev/null
buildah config \
  --env PYTHONPATH=/opt/py \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env VLLM_PLUGINS= \
  --env HF_HOME=/hf \
  --env MODEL=meta-llama/Llama-3.1-8B-Instruct \
  --env PD_SHM_ROOT=/dev/shm/rngd_kv_cluster \
  --workingdir /app \
  --entrypoint '["/venv/bin/python","/app/serving/pd_cluster.py"]' \
  --cmd '["--role","prefill","--chip","0","--port","8000"]' "$c"
buildah commit --rm "$c" "rngd-pd-prefill:$TAG"

buildah images | grep -E "rngd-pd-(decode|prefill)"
echo "BUILD_DONE"
