import sys,pathlib,subprocess,json
r=pathlib.Path("/home/gspark/workspace/codex_resident_delivery_20261102")
with (r/'evidence/cpu-causal-fixture02.log').open('x') as log:
 code=subprocess.call([sys.executable,str(r/'cpu-causal/causal.py'),'fixture','--onggi',str(r/'bin/onggi'),'--output',str(r/'cpu-causal-fixture02')],stdout=log,stderr=log)
assert code==0,'See retained fixture02 log'
d=json.loads((r/'cpu-causal-fixture02/fixture-result.json').read_text());assert d['valid']
print(json.dumps(dict(valid=d['valid'],plain_write_bytes=d['plain_write_bytes'],consume_write_bytes=d['consume_write_bytes'])))
