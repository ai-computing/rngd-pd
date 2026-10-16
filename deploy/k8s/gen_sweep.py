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
NODE = "npu-node-1"
APP = "llama8b-pd"                      # the InferencePool selector
ART = "/artifacts"
SVC = f"svc.cluster.local:8000"

HOST_MOUNTS = [
    ("local-pkgs", "/home/user/.local/lib/python3.10/site-packages",
     "/home/user/.local/lib/python3.10/site-packages"),
    ("sys-local-pkgs", "/host-sys-local-pkgs", "/usr/local/lib/python3.10/dist-packages"),
    ("sys-pkgs", "/host-sys-pkgs", "/usr/lib/python3/dist-packages"),
    ("libs", "/host-libs", "/usr/lib/x86_64-linux-gnu"),
    ("rbln-lib", "/host-rbln-lib", "/usr/lib"),
    ("rbln-bin", "/host-rbln-bin", "/usr/bin"),
]
APP_DIR = "/home/user/workspace/01_pd_disag/rngd_llmd"

APT = ("set -eu; export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; "
       "apt-get install -y -qq --no-install-recommends python3.10 python3.10-dev "
       "python3-pip libdrm2 libnuma1 libgomp1 ca-certificates tzdata >/dev/null; ")

BASE_ENV = [
    # /app/src first: the package is imported, not path-hacked into place.
    ("PYTHONPATH", "/app/src:/home/user/.local/lib/python3.10/site-packages:"
                   "/host-sys-local-pkgs:/host-sys-pkgs"),
    ("LD_LIBRARY_PATH", "/host-rbln-lib:/host-libs"),
    ("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/host-rbln-bin"),
    ("PYTHONUNBUFFERED", "1"),
    ("VLLM_PLUGINS", ""),
    ("HF_HOME", "/hf"),
    ("MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    ("PD_SHM_ROOT", "/dev/shm/rngd_kv_cluster"),
    ("PD_PREFETCH", os.environ.get("PD_PREFETCH", "1")),
    # FCConfig leaves max_num_unique_shapes unbounded and this repo never set
    # it, which is what actually capped the decode graph at two layers. Capping
    # it at 8 lifts the group to 8 and makes the kernel better as well:
    # 27.00 -> 20.10 ms TPOT through the gateway, 532 -> 792 GB/s in the kernel,
    # CPU fp32 reference PASS. 1 is far too tight (60.19 ms/step).
    # The graph has to be baked on the host first -- see bench/prebake.py. A pod
    # cannot compile it, and the worker drops back to group=2 if the artifact
    # is missing rather than crash-looping.
    ("RNGD_MAX_SHAPES", os.environ.get("RNGD_MAX_SHAPES", "8")),
    # How long a decode worker waits for a layer's KV. The default five
    # seconds only covers the transfer; it has to cover the prefill queue
    # ahead of it, which grows with the decode batch width.
    ("PD_KV_PULL_TIMEOUT", os.environ.get("PD_KV_PULL_TIMEOUT", "120")),
    ("RNGD_BUCKETS", os.environ.get("RNGD_BUCKETS", "")),
]

# Which decode scheduler the batched Furiosa worker runs. "static" holds a
# batch start to finish; "continuous" re-forms it every step.
RNGD_SCHED = os.environ.get("RNGD_SCHED", "static")
RNGD_ADMIT_MIN = os.environ.get("RNGD_ADMIT_MIN", "2")
# Layers per compiled decode module. Anything above 2 needs RNGD_MAX_SHAPES and
# a host-side bake; the worker falls back to 2 when the artifact is not there.
RNGD_GROUP = os.environ.get("RNGD_GROUP", "8")
ATOM_C = os.environ.get("ATOM_C", "")         # ATOM prefill width; "" = S
# The RNGD prefill graph is compiled for one width and a request pays all of
# it whatever its prompt: 88 ms at C=512 for a 16-token prompt as much as for
# a 448-token one. Separating the knob from S is the first step toward serving
# several widths and routing each prompt to the smallest that fits.
RNGD_C = os.environ.get("RNGD_C", "")
# Extra prefill widths ride in BASE_ENV as RNGD_BUCKETS, read straight from the
# environment there: BASE_ENV is built above this line, and a name defined
# here would not exist yet when that list is evaluated.
# Context width every graph is compiled for. Raising it means rebuilding every
# artifact and doubling the KV that crosses per request, so it is one knob.
S = os.environ.get("PD_S", "256")
# The control for the whole study: one card doing prefill and decode with the
# cache never leaving it. Only three RNGD cards are usable here (npu3 has an
# EDF loader fault), so it takes the batched decoder's card rather than adding
# to the fleet — the two are mutually exclusive.
# Decode widths the Furiosa side serves, one pod per width because the graph
# shape is fixed at compile time. B is the only lever on decode efficiency:
# a step reads every weight once and does B tokens with them, so its
# arithmetic intensity -- and therefore its MFU ceiling -- is linear in B.
RNGD_WIDTHS = [int(v) for v in os.environ.get("RNGD_WIDTHS", "1,4").split(",")]
# How many prefill cards stand behind the decode pool. A prefill card runs near
# 30% of peak and a decode card is held under 3% by bandwidth, so the ratio is
# what sets system-level efficiency -- but a second prefill card only earns its
# place when the prompt stream is heavy enough to keep it busy.
PREFILL_N = int(os.environ.get("PD_PREFILL_N", "1"))
SOLO = os.environ.get("PD_SOLO", "")
# PD_SOLO replaces the batched decoder with the one-card control, which is the
# right shape for an apples-to-apples efficiency number but makes the two
# impossible to load at the same time. PD_SOLO_EXTRA keeps both up on the three
# cards -- prefill, a decode pod, and solo -- so one load generator can hit
# them under identical conditions instead of across two deployments minutes
# apart. That matters here: the claim being tested is about behaviour under
# concurrent load, and load is not reproducible across a redeploy.
SOLO_EXTRA = os.environ.get("PD_SOLO_EXTRA", "")
# Three usable RNGD cards, one pod each. Asking for more prefill cards than
# that leaves over is a scheduling deadlock that shows up as Pending pods
# twenty minutes later, so it is worth refusing here.
RNGD_CARDS = int(os.environ.get("RNGD_CARDS", "3"))


def rngd_decode_name(B):
    return "sw-rngd-decode" if B == 1 else f"sw-rngd-decode-b{B}"
SOLO_B = os.environ.get("PD_SOLO_B", "4")
ATOM_C = ATOM_C or S               # one pass means width == context
# "_lm" artifacts carry lm_head on the card. The card is 16.88 GB and the
# artifact grows 14.2 -> 15.2 GB, which fits; off-card it cost a 2.1 GB host
# read per step and that cost scaled with batch width.
ATOM_LM = os.environ.get("ATOM_LM", "_lm")

# name, vendor, args, extra env, memory
WORKERS = [
    (f"sw-rngd-prefill{'' if i == 0 else i}", "rngd",
     f"--role prefill --backend rngd-full --C {RNGD_C or S} --group 2 "
     f"--chip {i} --S {S}",
     [], "64Gi")
    for i in range(PREFILL_N)
] + ([
    # One card holds prefill and decode with the cache never leaving it: the
    # control the whole study needs, and mutually exclusive with the batched
    # decoder because only three RNGD cards are usable here.
    ("sw-rngd-solo", "rngd",
     f"--role solo --C {S} --group 2 --decode-group {RNGD_GROUP} "
     f"--B {SOLO_B} --batch-wait 8000 --chip {PREFILL_N} --S {S}", [], "96Gi"),
] if SOLO else ([
    # Same decode group as the disaggregated pod. Leaving solo on group=2
    # while the PD decoder ran group=8 handed the comparison a confound worth
    # about 9 ms a step -- solo's TPOT came out 33 ms against PD's 22 and it
    # read as prefill interference, when most of it was just a worse kernel.
    ("sw-rngd-solo", "rngd",
     f"--role solo --C {S} --group 2 --decode-group {RNGD_GROUP} "
     f"--B {SOLO_B} --batch-wait 8000 --sched {RNGD_SCHED} "
     f"--admit-min {RNGD_ADMIT_MIN} "
     f"--chip {PREFILL_N + len(RNGD_WIDTHS)} --S {S}", [], "96Gi"),
] if SOLO_EXTRA else []) + [
    (rngd_decode_name(B), "rngd",
     (f"--role decode --backend rngd --chip {PREFILL_N + i} --S {S}" if B == 1 else
      f"--role decode --backend rngd --B {B} --group {RNGD_GROUP} "
      f"--batch-wait 8000 --sched {RNGD_SCHED} --admit-min {RNGD_ADMIT_MIN} "
      f"--chip {PREFILL_N + i} --S {S}"),
     [], "96Gi")
    for i, B in enumerate(RNGD_WIDTHS)
]) + [
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
        f"--role decode --backend atom --B {b} --batch-wait 8000 "
        f"--sched {RNGD_SCHED} --admit-min {RNGD_ADMIT_MIN} --S {S}",
        [("RBLN_BATCH_ARTIFACT", f"{ART}/decode_S{S}_B{b}{ATOM_LM}.rbln"),
         ("RBLN_TABLES", f"{ART}/tables.npz")], "48Gi"))


def url(name):
    return f"http://{name}.{NS}.{SVC}"


RNGD_DEC = ({} if SOLO else
            {str(B): url(rngd_decode_name(B)) for B in RNGD_WIDTHS})

# One URL when a single card prefills, a list when several do — the gateway
# round-robins a list. Keeping the single case a bare string means every
# existing config and every stored ConfigMap keeps its exact shape.
RNGD_PRE = (url("sw-rngd-prefill") if PREFILL_N == 1 else
            [url(f"sw-rngd-prefill{'' if i == 0 else i}") for i in range(PREFILL_N)])

if PREFILL_N + (0 if SOLO else len(RNGD_WIDTHS)) + (1 if SOLO_EXTRA else 0) > RNGD_CARDS:
    raise SystemExit(
        f"RNGD 카드 {RNGD_CARDS}장인데 프리필 {PREFILL_N} + 디코드 "
        f"{len(RNGD_WIDTHS)} 를 요구함 — PD_PREFILL_N 이나 RNGD_WIDTHS 를 줄일 것")

TARGETS = {
    "llama8b-rngd-alone": {"prefill": RNGD_PRE,
                           "decode": dict(RNGD_DEC)},
    "llama8b-pd-forward": {"prefill": RNGD_PRE,
                           "decode": {str(b): url(f"sw-atom-decode-b{b}")
                                      for b in (1, 2, 4)}},
    "llama8b-pd-reverse": {"prefill": url("sw-atom-prefill"),
                           "decode": dict(RNGD_DEC)},
    "llama8b-atom-alone": {"prefill": url("sw-atom-prefill"),
                           "decode": {str(b): url(f"sw-atom-decode-b{b}")
                                      for b in (1, 2, 4)}},
}
if SOLO or SOLO_EXTRA:
    TARGETS["llama8b-solo"] = {"solo": url("sw-rngd-solo")}


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
            "      - {name: hf, hostPath: {path: /home/user/.cache/huggingface, type: Directory}}",
            "      - {name: dshm, hostPath: {path: /dev/shm, type: Directory}}"]
    if vendor == "rngd":
        out += ["      - {name: venv, hostPath: {path: /home/user/venvs/furiosa, type: Directory}}",
                "      - {name: devrngd, hostPath: {path: /dev/rngd, type: Directory}}"]
    else:
        out += ["      - {name: artifacts, hostPath: {path: /home/user/rbln-llama8b-pd, type: Directory}}"]
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
{volumes("atom").replace('      - {name: artifacts, hostPath: {path: /home/user/rbln-llama8b-pd, type: Directory}}', '      - {name: cfg, configMap: {name: pd-sweep-targets}}')}
"""


if __name__ == "__main__":
    print(__doc__.split("\n")[0].join(["# ", "\n"]), end="")
    for w in WORKERS:
        print(worker(*w), end="")
    print(router(), end="")
