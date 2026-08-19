#!/usr/bin/env python3
"""Emit the sweep deployment: six workers, one router, four configurations.

The four cases the sweep compares differ only in which card prefills and which
decodes, so they are built from a shared pool of workers rather than four
separate stacks:

  rngd_alone   RNGD kv-free prefill  -> RNGD decode
  pd_forward   RNGD kv-free prefill  -> ATOM decode   (B = 1, 2, 4)
  pd_reverse   ATOM chunked prefill  -> RNGD decode
  atom_alone   ATOM chunked prefill  -> ATOM decode   (B = 1, 2, 4)

Batch width needs its own pod per width: an ATOM decode artifact is 14 GB and
the graph shape is fixed at compile time, so one card cannot hold B=1, 2 and 4
at once. That costs three of the four ATOM cards; the fourth prefills.

The router carries the InferencePool's `app` label, so every request still
arrives through the cluster gateway → InferencePool → EPP path. It picks the
pair from the request's model name and the decode width from a `batch` field.

  python3 gen_sweep.py > sweep-matrix.yaml && kubectl apply -f sweep-matrix.yaml
"""
import json
import os

NS = "llm-serving"
NODE = "etri-001"
APP = "llama8b-pd"                      # the InferencePool selector
ART = "/artifacts"
SVC = f"svc.cluster.local:8000"

HOST_MOUNTS = [
    ("local-pkgs", "/home/gspark/.local/lib/python3.10/site-packages",
     "/home/gspark/.local/lib/python3.10/site-packages"),
    ("sys-local-pkgs", "/host-sys-local-pkgs", "/usr/local/lib/python3.10/dist-packages"),
    ("sys-pkgs", "/host-sys-pkgs", "/usr/lib/python3/dist-packages"),
    ("libs", "/host-libs", "/usr/lib/x86_64-linux-gnu"),
    ("rbln-lib", "/host-rbln-lib", "/usr/lib"),
    ("rbln-bin", "/host-rbln-bin", "/usr/bin"),
]
APP_DIR = "/home/gspark/workspace/01_pd_disag/rngd_llmd"

APT = ("set -eu; export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; "
       "apt-get install -y -qq --no-install-recommends python3.10 python3.10-dev "
       "python3-pip libdrm2 libnuma1 libgomp1 ca-certificates tzdata >/dev/null; ")

BASE_ENV = [
    # /app/src first: the package is imported, not path-hacked into place.
    ("PYTHONPATH", "/app/src:/home/gspark/.local/lib/python3.10/site-packages:"
                   "/host-sys-local-pkgs:/host-sys-pkgs"),
    ("LD_LIBRARY_PATH", "/host-rbln-lib:/host-libs"),
    ("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/host-rbln-bin"),
    ("PYTHONUNBUFFERED", "1"),
    ("VLLM_PLUGINS", ""),
    ("HF_HOME", "/hf"),
    ("MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    ("PD_SHM_ROOT", "/dev/shm/rngd_kv_cluster"),
]

# Which decode scheduler the batched Furiosa worker runs. "static" holds a
# batch start to finish; "continuous" re-forms it every step.
RNGD_SCHED = os.environ.get("RNGD_SCHED", "static")
RNGD_ADMIT_MIN = os.environ.get("RNGD_ADMIT_MIN", "2")
ATOM_C = os.environ.get("ATOM_C", "")         # ATOM prefill width; "" = S
# Context width every graph is compiled for. Raising it means rebuilding every
# artifact and doubling the KV that crosses per request, so it is one knob.
S = os.environ.get("PD_S", "256")
ATOM_C = ATOM_C or S               # one pass means width == context

# name, vendor, args, extra env, memory
WORKERS = [
    ("sw-rngd-prefill", "rngd",
     f"--role prefill --backend rngd-full --C {S} --group 2 --chip 0 --S {S}", [], "64Gi"),
    ("sw-rngd-decode", "rngd",
     f"--role decode --backend rngd --chip 1 --S {S}", [], "64Gi"),
    # Batched Furiosa decode. The 32-layer batched graph does not compile, so
    # this reuses a two-layer module sixteen times — the step is B-invariant,
    # which is what makes the reuse worth its dispatch cost.
    ("sw-rngd-decode-b4", "rngd",
     "--role decode --backend rngd --B 4 --group 2 --batch-wait 8000 "
     f"--sched {RNGD_SCHED} --admit-min {RNGD_ADMIT_MIN} --chip 2 --S {S}", [], "64Gi"),
    # Chunk width is what makes ATOM prefill linear in prompt length: at C=16
    # the weights are re-read P/16 times. C=128 cuts that eightfold.
    ("sw-atom-prefill", "atom",
     f"--role prefill --backend atom --C {ATOM_C} --S {S}",
     [("RBLN_PREFILL_ARTIFACT", f"{ART}/prefill_S{S}_C{ATOM_C}.rbln"),
      ("RBLN_TABLES", f"{ART}/tables.npz")], "48Gi"),
]
for b in (1, 2, 4):
    WORKERS.append((
        f"sw-atom-decode-b{b}", "atom",
        f"--role decode --backend atom --B {b} --batch-wait 8000 --S {S}",
        [("RBLN_BATCH_ARTIFACT", f"{ART}/decode_S{S}_B{b}.rbln"),
         ("RBLN_TABLES", f"{ART}/tables.npz")], "48Gi"))


def url(name):
    return f"http://{name}.{NS}.{SVC}"


RNGD_DEC = {"1": url("sw-rngd-decode"), "4": url("sw-rngd-decode-b4")}

TARGETS = {
    "llama8b-rngd-alone": {"prefill": url("sw-rngd-prefill"),
                           "decode": dict(RNGD_DEC)},
    "llama8b-pd-forward": {"prefill": url("sw-rngd-prefill"),
                           "decode": {str(b): url(f"sw-atom-decode-b{b}")
                                      for b in (1, 2, 4)}},
    "llama8b-pd-reverse": {"prefill": url("sw-atom-prefill"),
                           "decode": dict(RNGD_DEC)},
    "llama8b-atom-alone": {"prefill": url("sw-atom-prefill"),
                           "decode": {str(b): url(f"sw-atom-decode-b{b}")
                                      for b in (1, 2, 4)}},
}


def env_block(pairs, indent="        "):
    return "\n".join(f'{indent}- {{name: {k}, value: "{v}"}}' for k, v in pairs)


def mounts(vendor):
    out = [f"        - {{name: {n}, mountPath: {mp}, readOnly: true}}"
           for n, mp, _ in HOST_MOUNTS]
    out += ["        - {name: app, mountPath: /app}",
            "        - {name: hf, mountPath: /hf}",
            "        - {name: dshm, mountPath: /dev/shm}"]
    if vendor == "rngd":
        out += ["        - {name: venv, mountPath: /venv, readOnly: true}",
                "        - {name: devrngd, mountPath: /dev/rngd}"]
    else:
        out += ["        - {name: artifacts, mountPath: /artifacts, readOnly: true}"]
    return "\n".join(out)


def volumes(vendor):
    out = [f"      - {{name: {n}, hostPath: {{path: {hp}, type: Directory}}}}"
           for n, _, hp in HOST_MOUNTS]
    out += [f"      - {{name: app, hostPath: {{path: {APP_DIR}, type: Directory}}}}",
            "      - {name: hf, hostPath: {path: /home/gspark/.cache/huggingface, type: Directory}}",
            "      - {name: dshm, hostPath: {path: /dev/shm, type: Directory}}"]
    if vendor == "rngd":
        out += ["      - {name: venv, hostPath: {path: /home/gspark/venvs/furiosa, type: Directory}}",
                "      - {name: devrngd, hostPath: {path: /dev/rngd, type: Directory}}"]
    else:
        out += ["      - {name: artifacts, hostPath: {path: /home/gspark/rbln-llama8b-pd, type: Directory}}"]
    return "\n".join(out)


def worker(name, vendor, args, extra_env, mem):
    py = "/venv/bin/python" if vendor == "rngd" else "python3.10"
    res = "furiosa.ai/rngd" if vendor == "rngd" else "rebellions.ai/ATOM"
    # RNGD needs privileged: the device plugin does not hand over npuNp2pmem,
    # and an 8-PE Mesh(2) device cannot be opened without it.
    priv = "\n          privileged: true" if vendor == "rngd" else ""
    return f"""---
apiVersion: v1
kind: Service
metadata: {{name: {name}, namespace: {NS}}}
spec:
  selector: {{app: {name}}}
  ports: [{{port: 8000, targetPort: 8000}}]
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  namespace: {NS}
  labels: {{app: {name}, llm-d.ai/sweep: "true"}}
spec:
  replicas: 1
  strategy: {{type: Recreate}}      # one card per pod: no two may overlap
  selector: {{matchLabels: {{app: {name}}}}}
  template:
    metadata:
      labels: {{app: {name}, llm-d.ai/sweep: "true"}}
    spec:
      nodeSelector: {{kubernetes.io/hostname: {NODE}}}
      containers:
      - name: worker
        image: ubuntu:22.04
        imagePullPolicy: IfNotPresent
        command: [bash, -c]
        args:
        - |
          {APT}exec {py} -m rngd_pd.serving.cluster {args} --port 8000
        env:
{env_block(BASE_ENV + extra_env)}
        ports: [{{containerPort: 8000}}]
        resources:
          limits: {{{res}: "1"}}
          requests: {{cpu: "8", memory: {mem}, {res}: "1"}}
        readinessProbe:
          httpGet: {{path: /health, port: 8000}}
          initialDelaySeconds: 20
          periodSeconds: 10
          failureThreshold: 120         # cold start: weights, then artifact load
        securityContext:
          seccompProfile: {{type: Unconfined}}{priv}
        volumeMounts:
{mounts(vendor)}
      volumes:
{volumes(vendor)}
"""


def indent(text, pad):
    return "\n".join(pad + line for line in text.splitlines())


def router():
    targets = indent(json.dumps(TARGETS, indent=2), "    ")
    return f"""---
apiVersion: v1
kind: ConfigMap
metadata: {{name: pd-sweep-targets, namespace: {NS}}}
data:
  targets.json: |
{targets}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hetero-pd-gateway
  namespace: {NS}
  labels: {{app: {APP}, llm-d.ai/role: pd-gateway}}
spec:
  replicas: 1
  selector: {{matchLabels: {{app: {APP}}}}}
  template:
    metadata:
      labels: {{app: {APP}, llm-d.ai/role: pd-gateway}}
    spec:
      nodeSelector: {{kubernetes.io/hostname: {NODE}}}
      containers:
      - name: gateway
        image: ubuntu:22.04
        imagePullPolicy: IfNotPresent
        command: [bash, -c]
        args:
        - |
          {APT}exec python3.10 -m rngd_pd.serving.cluster \\
            --role gateway --S {S} --port 8000 --targets-file /cfg/targets.json
        env:
{env_block([e for e in BASE_ENV if e[0] != "VLLM_PLUGINS"])}
        ports: [{{containerPort: 8000}}]
        resources: {{requests: {{cpu: "4", memory: 8Gi}}}}
        readinessProbe:
          httpGet: {{path: /health, port: 8000}}
          initialDelaySeconds: 10
          periodSeconds: 10
          failureThreshold: 60
        securityContext:
          seccompProfile: {{type: Unconfined}}
        volumeMounts:
{mounts("atom").replace('        - {name: artifacts, mountPath: /artifacts, readOnly: true}', '        - {name: cfg, mountPath: /cfg, readOnly: true}')}
      volumes:
{volumes("atom").replace('      - {name: artifacts, hostPath: {path: /home/gspark/rbln-llama8b-pd, type: Directory}}', '      - {name: cfg, configMap: {name: pd-sweep-targets}}')}
"""


if __name__ == "__main__":
    print(__doc__.split("\n")[0].join(["# ", "\n"]), end="")
    for w in WORKERS:
        print(worker(*w), end="")
    print(router(), end="")
