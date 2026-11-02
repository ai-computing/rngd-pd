set -eu
python3 - <<'PY'
import pathlib,json,subprocess
p=pathlib.Path('/home/gspark/workspace/codex_binding_conformance_20261030/atom-runtime-source')
if not p.exists():
 subprocess.run(['git','clone','--single-branch','--branch','exp/37-atom-binding','https://github.com/ai-computing/rngd-pd.git',str(p)],capture_output=True,check=True,timeout=70)
 subprocess.run(['git','-C',str(p),'checkout','--detach','fbfdb8bb9aa86750f144fa4ead276fe17f6db537'],capture_output=True,check=True)
head=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip();assert head=='fbfdb8bb9aa86750f144fa4ead276fe17f6db537'
print(json.dumps(dict(head=head)))

PY
