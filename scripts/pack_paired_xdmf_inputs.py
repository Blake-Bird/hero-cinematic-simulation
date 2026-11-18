#!/usr/bin/env python3
"""Losslessly compress portable ParaView HDF5 inputs, checking every dataset byte."""
import argparse,json,os,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.foundation import file_sha256,atomic_json

def repack(job):
 path,expected,receipt=Path(job[0]),job[1],Path(job[2])
 if receipt.exists():
  row=json.loads(receipt.read_text());current=file_sha256(path)
  if current==row['sha256']:return row
  stage=path.with_suffix('.packing.h5')
  if current==expected and stage.exists() and file_sha256(stage)==row['sha256']:stage.replace(path);return row
  receipt.unlink()
 before=file_sha256(path)
 if before!=expected:raise ValueError(f'Changed XDMF input {path}')
 old_size=path.stat().st_size;stage=path.with_suffix('.packing.h5');datasets=[]
 try:
  with h5py.File(path,'r') as src,h5py.File(stage,'w') as dst:
   def group(a,b):
    for k,v in a.attrs.items():b.attrs[k]=v
    for name,node in a.items():
     if isinstance(node,h5py.Group):group(node,b.create_group(name));continue
     options={'compression':'gzip','compression_opts':1,'shuffle':True} if node.shape and all(node.shape) else {}
     d=b.create_dataset(name,shape=node.shape,dtype=node.dtype,**options)
     for k,v in node.attrs.items():d.attrs[k]=v
     if node.shape:
      for start in range(0,node.shape[0],65536):d[start:start+65536]=node[start:start+65536]
     else:d[()]=node[()]
   group(src,dst)
  with h5py.File(path,'r') as src,h5py.File(stage,'r') as dst:
   def check(a,b):
    if set(a.keys())!=set(b.keys()) or set(a.attrs)!=set(b.attrs):raise ValueError('HDF5 tree mismatch')
    for k in a.attrs:
     if not np.array_equal(a.attrs[k],b.attrs[k]):raise ValueError('HDF5 attribute mismatch')
    for name,node in a.items():
     other=b[name]
     if isinstance(node,h5py.Group):check(node,other);continue
     if node.shape!=other.shape or node.dtype!=other.dtype or set(node.attrs)!=set(other.attrs):raise ValueError('HDF5 dataset metadata mismatch')
     for k in node.attrs:
      if not np.array_equal(node.attrs[k],other.attrs[k]):raise ValueError('HDF5 dataset attribute mismatch')
     if node.shape:
      for start in range(0,node.shape[0],65536):
       if node[start:start+65536].tobytes()!=other[start:start+65536].tobytes():raise ValueError('HDF5 payload changed')
     elif node[()].tobytes()!=other[()].tobytes():raise ValueError('HDF5 scalar changed')
     datasets.append({'path':node.name,'shape':list(node.shape),'dtype':str(node.dtype)})
   check(src,dst)
  after=file_sha256(stage);row={'path':str(path),'before_sha256':before,'sha256':after,'before_bytes':old_size,'bytes':stage.stat().st_size,'numeric_payload':'BITWISE_IDENTICAL','datasets':datasets}
  # The verified receipt is recoverable if a later packing job is interrupted.
  atomic_json(receipt,row);stage.replace(path);return row
 except BaseException:
  if stage.exists():stage.unlink()
  raise

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('atlas',type=Path);args=parser.parse_args();root=args.atlas.resolve();provenance=root/'Source/Storage_cleanup';provenance.mkdir(exist_ok=True);audit=root/'07_Status_and_Provenance/lossless_xdmf_packing.json';rows=json.loads(audit.read_text())['files'] if audit.exists() else []
 for model in ('EDE','LCDM'):
  folder=root/'05_Blender_and_3D'/model/f'{model}_z_199_to_000_ParaView_XDMF';contract=folder/'xdmf_contract.json';record=json.loads(contract.read_text())
  if record.get('storage_packing',{}).get('state')=='LOSSLESS_VERIFIED':print('ALREADY_PACKED',model,flush=True);continue
  backup=provenance/f'{model}_xdmf_contract_before_packing.json'
  if not backup.exists():backup.write_bytes(contract.read_bytes())
  previous=json.loads(backup.read_text());receipts=provenance/(model+'_packing_receipts');receipts.mkdir(exist_ok=True)
  jobs=[(str(folder/item['path']),item['sha256'],str(receipts/(Path(item['path']).name+'.json'))) for item in previous['inputs']]
  completed=[]
  with ProcessPoolExecutor(max_workers=4) as pool:
   for future in as_completed([pool.submit(repack,job) for job in jobs]):
    completed.append(future.result())
    if len(completed)%80==0:print('PACKED',model,len(completed),'/',len(jobs),flush=True)
  by_path={row['path']:row for row in completed}
  for item in record['inputs']:
   row=by_path[str(folder/item['path'])];item['original_export_sha256']=row['before_sha256'];item['sha256']=row['sha256']
  record['storage_packing']={'state':'LOSSLESS_VERIFIED','compression':'HDF5 gzip level 1 with shuffle; shapes, types, attributes and every numeric byte verified before replacement.','original_export_contract':str(backup.relative_to(root)),'verification':'../../../07_Status_and_Provenance/lossless_xdmf_packing.json','algorithm_sha256':file_sha256(__file__)}
  atomic_json(contract,record)
  for row in completed:row['path']=str(Path(row['path']).relative_to(root));row['model']=model
  rows.extend(completed);atomic_json(root/'07_Status_and_Provenance/lossless_xdmf_packing.json',{'status':'PASS','files':rows,'before_bytes':sum(r['before_bytes'] for r in rows),'bytes':sum(r['bytes'] for r in rows)})
  import shutil
  shutil.rmtree(receipts)
  print('MODEL_PACKED',model,'saved GiB',sum(r['before_bytes']-r['bytes'] for r in completed)/2**30,flush=True)
if __name__=='__main__':main()
