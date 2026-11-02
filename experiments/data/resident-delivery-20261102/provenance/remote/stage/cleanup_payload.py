"""Remove only exported, completed task payloads; retained JSON remains local/in archives."""
import base64,gzip,hashlib,json,pathlib,shutil,sys
ROOT=pathlib.Path('/home/gspark/workspace/codex_resident_delivery_20261102')
cohort=sys.argv[1];assert cohort.replace('-','').isalnum()
packed=ROOT/'evidence'/(cohort+'-small.json.gz');archive=json.loads(gzip.decompress(packed.read_bytes()))
assert archive['cohort']==cohort
native=ROOT/'runs'/(cohort+'-cohort.json');record=json.loads(native.read_text());assert record['status']=='completed'
targets=[ROOT/'prep'/cohort]+[ROOT/'runs'/t['trial_id']/'work' for t in record['trials']]
for p in targets:
 assert p.resolve().is_relative_to(ROOT.resolve()) and p.resolve()!=ROOT.resolve() and not p.is_symlink()
 if p.exists():
  assert not p.is_mount(),'Owned mount still live'
  for child in p.rglob('*'):
   assert not child.is_symlink(),'Refuse symlinked payload cleanup'
for f in archive['files']:
 p=ROOT/f['path']
 if any(p==t or p.is_relative_to(t) for t in targets):
  b=p.read_bytes();assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256'],'exported source changed before cleanup'
before=shutil.disk_usage(ROOT).free
for p in targets:
 if p.exists():shutil.rmtree(p)
receipt=dict(cohort=cohort,export_archive_sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),removed=[str(p) for p in targets],free_before=before,free_after=shutil.disk_usage(ROOT).free,npu_requested=0)
out=ROOT/'evidence'/(cohort+'-payload-cleanup.json');assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
