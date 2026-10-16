"""Furiosa 2026.4.0 experimental TCP connector codec.

Observed and checked against real native/native transfers; this is NOT a stable
vendor API. rkyv archives: 36-byte outer enum, little endian, relative pointers.
Supported topology: one RNGD, BF16 Llama KV, no auxiliary/sliding-window cache.
"""
import json,struct
import numpy as np
ENTRY_DTYPE=np.dtype([(name,'<u4') for name in ('tier','tensor','block','bytes')])
ROOT_SIZE=36
MAX_HEADER=1<<20
class ProtocolError(ValueError):pass
def u32(b,o):return struct.unpack_from('<I',b,o)[0]
def i32(b,o):return struct.unpack_from('<i',b,o)[0]
def _span(b,start,size):
 if start<0 or size<0 or start+size>len(b):raise ProtocolError('archive pointer outside header')
 return b[start:start+size]
def string(b,o):
 _span(b,o,8)
 if b[o]&128:
  n=(b[o]&63)+(int.from_bytes(b[o+1:o+4],'little')<<6)
  raw=_span(b,o+i32(b,o+4),n)
 else:raw=b[o:o+8].split(b'\xff',1)[0]
 return raw.decode('utf-8')
def _str(buf,pos,value):
 raw=value.encode('utf8')
 if len(raw)<=8:buf[pos:pos+8]=raw+b'\xff'*(8-len(raw));return
 raise ProtocolError('use builder for outlined strings')
def _archive(fields,control=None,entries=None):
 # fields = strings in order; append external strings before the fixed root.
 prefix=bytearray();refs=[]
 for value in fields:
  raw=value.encode();refs.append((len(prefix),raw));prefix.extend(raw if len(raw)>8 else b'')
 while len(prefix)%4:prefix.append(0)
 entry_start=len(prefix)
 if entries is not None:
  for x in entries:prefix.extend(struct.pack('<I',x))
 root=len(prefix);prefix.extend(bytes(ROOT_SIZE))
 struct.pack_into('<I',prefix,root,0)
 if control is not None:struct.pack_into('<I',prefix,root+4,control)
 for idx,(off,raw) in enumerate(refs):
  pos=root+8+idx*8
  if len(raw)<=8:prefix[pos:pos+8]=raw+b'\xff'*(8-len(raw))
  else:
   packed=(len(raw)&63)|128|((len(raw)>>6)<<8)
   struct.pack_into('<Ii',prefix,pos,packed,off-pos)
 return prefix,root,entry_start
def control(tag,request_id):
 b,_,_=_archive([request_id],tag);return bytes(b)
def handshake(endpoint,layout):
 b,_,_=_archive([endpoint,json.dumps(layout,separators=(',',':')),'null'],0);return bytes(b)
def pull(request_id,blocks,start_pos=0):
 if not blocks or len(blocks)>512 or any(x<0 or x>=512 for x in blocks):raise ProtocolError('invalid host block range')
 b,r,e=_archive([request_id],2,blocks)
 struct.pack_into('<IiIiI',b,r+16,start_pos,e-(r+20),len(blocks),e+len(blocks)*4-(r+28),0)
 return bytes(b)
def parse(b, array_entries=False):
 if len(b)<ROOT_SIZE or len(b)>MAX_HEADER or len(b)%4:raise ProtocolError('invalid archive size')
 r=len(b)-ROOT_SIZE;kind=u32(b,r)
 if kind==0:
  tag=u32(b,r+4)
  if tag==0:return dict(kind='handshake',endpoint=string(b,r+8),layout=json.loads(string(b,r+16)),aux=json.loads(string(b,r+24)))
  if tag not in [1,2,3,4]:raise ProtocolError('unsupported control tag')
  out=dict(kind={1:'ready',2:'pull',3:'ack',4:'cancel'}[tag],request_id=string(b,r+8))
  if tag==2:
   n=u32(b,r+24);p=r+20+i32(b,r+20)
   if n>512:raise ProtocolError('too many blocks')
   out.update(start_pos=u32(b,r+16),blocks=list(struct.unpack('<'+'I'*n,_span(b,p,n*4))))
   if u32(b,r+32):raise ProtocolError('auxiliary KV unsupported')
  return out
 if kind!=1:raise ProtocolError('unsupported outer tag')
 n=u32(b,r+16);p=r+12+i32(b,r+12)
 if n>64*512:raise ProtocolError('too many KV entries')
 values=_span(memoryview(b),p,n*16)
 entries=np.frombuffer(values,dtype=ENTRY_DTYPE,count=n) if array_entries else [dict(zip(['tier','tensor','block','bytes'],x)) for x in struct.iter_unpack('<IIII',values)]
 if b[r+20] not in (0,1):raise ProtocolError('invalid final flag')
 return dict(kind='kv',request_id=string(b,r+4),entries=entries,is_final=bool(b[r+20]))
