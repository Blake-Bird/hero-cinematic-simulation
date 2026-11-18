"""Loopback-only optional-lab UI with fixed, bounded run commands.

The server never accepts arbitrary commands. Source viewing is restricted to
inventoried first-party Python files. Artifacts are confined to owned job IDs.
A per-process token and exact Host/Origin checks protect run/cancel requests.
"""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs,unquote
import json
import hashlib
import mimetypes
import os
import secrets
import signal
import subprocess
import sys
import threading
import uuid

from optional_tests.catalogue import ROOT,HERE,build_catalogue,source_inventory


def confined(root,relative):
    """Resolve a nonempty relative path strictly inside its declared root."""
    root=Path(root).resolve();path=(root/relative).resolve()
    if not relative or Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError('Path escapes its declared root')
    return path


class JobStore:
    """One bounded subprocess at a time; preserve logs and cancellation evidence."""
    def __init__(self,root):
        self.root=Path(root).expanduser().resolve()
        if self.root.is_relative_to(ROOT):raise ValueError('Keep generated lab output outside the source tree')
        self.root.mkdir(parents=True,exist_ok=True)
        self.jobs={};self.lock=threading.RLock()

    def launch(self,profile,particles,seed):
        """Launch only the shipped runner with validated scalar arguments."""
        if profile not in {'quick','visual','native','libraries','combined','engines','physics','complete','figures'}:raise ValueError('Unknown profile')
        if type(particles) is not int or particles not in {32,64,512,1024,4096,8192}:raise ValueError('Choose a declared particle count')
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid uint32 seed')
        with self.lock:
            if any(job['process'].poll() is None for job in self.jobs.values()):raise RuntimeError('A lab run is already active')
            identifier=uuid.uuid4().hex;container=self.root/identifier;container.mkdir()
            output=container/'results';log=(container/'run.log').open('wb')
            command=[sys.executable,str(HERE/'lab.py'),'run','--profile',profile,'--particles',str(particles),
                     '--seed',str(seed),'--output',str(output)]
            try:process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            except Exception:log.close();raise
            self.jobs[identifier]={'process':process,'log':log,'container':container,'output':output,'cancelled':False}
            return identifier

    def snapshot(self,identifier):
        """Read current child progress; derive terminal failure from the actual process exit."""
        with self.lock:
            job=self.jobs.get(identifier)
            if job is None:raise KeyError('Unknown job in this server session')
            code=job['process'].poll();path=job['output']/'run.json'
            state=json.loads(path.read_text()) if path.exists() else {'status':'STARTING','steps':[],'artifacts':[]}
            if code is not None and not job['log'].closed:job['log'].close()
            if job['cancelled']:state['status']='CANCELLED'
            elif code is not None and state['status'] in {'RUNNING','STARTING'}:
                state.update(status='FAILED',reason='Runner stopped before a complete report; inspect run.log')
            state.update(job_id=identifier,process_running=code is None,exit_code=code)
            return state

    def cancel(self,identifier):
        """Stop the owned process group and record the user's cancellation."""
        with self.lock:
            job=self.jobs.get(identifier)
            if job is None:raise KeyError('Unknown job')
            if job['process'].poll() is None:
                os.killpg(job['process'].pid,signal.SIGTERM);job['cancelled']=True
                (job['container']/'cancelled.json').write_text(json.dumps({'status':'CANCELLED','source':'local UI action'}))
            return self.snapshot(identifier)


def make_handler(store,token,allowed_host):
    """Create an HTTP handler bound to one local authority and one job store."""
    allowed_sources={r['path'] for r in source_inventory()}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args):pass
        def send(self,status,body,kind='application/json'):
            if isinstance(body,(dict,list)):body=json.dumps(body,allow_nan=False).encode()
            if isinstance(body,str):body=body.encode()
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)));self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def authority(self,mutation=False):
            if self.headers.get('Host')!=allowed_host:raise PermissionError('Use the printed 127.0.0.1 URL')
            if mutation:
                origin=self.headers.get('Origin')
                if origin and origin!='http://'+allowed_host:raise PermissionError('Cross-origin request rejected')
                if not secrets.compare_digest(self.headers.get('X-Lab-Token',''),token):raise PermissionError('Missing lab token')
        def file(self,path):
            if not path.is_file():raise FileNotFoundError('Artifact not found')
            kind=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            if path.suffix in {'.py','.log','.md'}:kind='text/plain; charset=utf-8'
            self.send(200,path.read_bytes(),kind)
        def do_GET(self):
            try:
                self.authority();url=urlsplit(self.path);query=parse_qs(url.query);path=unquote(url.path)
                if path=='/api/bootstrap':self.send(200,{'token':token,'catalogue':build_catalogue()})
                elif path=='/api/job':self.send(200,store.snapshot(query['id'][0]))
                elif path=='/source':
                    name=query.get('path',[''])[0]
                    if name not in allowed_sources:raise PermissionError('Source is outside the first-party inventory')
                    start=max(1,min(int(query.get('line',['1'])[0]),100000))
                    raw=confined(ROOT,name).read_bytes();lines=raw.decode('utf-8').splitlines()
                    self.send(200,{'path':name,'first_line':start,'sha256':hashlib.sha256(raw).hexdigest(),
                        'text':'\n'.join(f'{i+1:5d}  {line}' for i,line in enumerate(lines) if start-1<=i<start+79)})
                elif path.startswith('/reference/'):
                    self.file(confined(HERE/'reference_gallery',path.removeprefix('/reference/')))
                elif path.startswith('/artifacts/'):
                    parts=path.removeprefix('/artifacts/').split('/',1)
                    if len(parts)!=2 or parts[0] not in store.jobs:raise PermissionError('Unknown owned artifact directory')
                    self.file(confined(store.jobs[parts[0]]['container'],parts[1]))
                elif path in {'/','/index.html'}:self.file(HERE/'app/index.html')
                elif path in {'/styles.css','/app.js'}:self.file(HERE/'app'/path[1:])
                else:raise FileNotFoundError('Unknown route')
            except PermissionError as error:self.send(403,{'error':str(error)})
            except (KeyError,ValueError,FileNotFoundError) as error:self.send(404,{'error':str(error)})
            except Exception as error:self.send(500,{'error':str(error)})
        def do_POST(self):
            try:
                self.authority(mutation=True)
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('Invalid request size')
                request=json.loads(self.rfile.read(length))
                if self.path=='/api/run':
                    if set(request)!={'profile','particles','seed'}:raise ValueError('Complete fixed run arguments required')
                    identifier=store.launch(**request);self.send(202,{'job_id':identifier})
                elif self.path=='/api/cancel':self.send(200,store.cancel(request['job_id']))
                else:raise ValueError('Unknown action')
            except PermissionError as error:self.send(403,{'error':str(error)})
            except RuntimeError as error:self.send(409,{'error':str(error)})
            except (KeyError,ValueError) as error:self.send(400,{'error':str(error)})
    return Handler


def serve(port,output_root):
    """Serve only on IPv4 loopback; printed URL is the required Host authority."""
    if type(port) is not int or not 1<=port<=65535:raise ValueError('Port must be in [1,65535]')
    store=JobStore(output_root);token=secrets.token_urlsafe(32)
    server=ThreadingHTTPServer(('127.0.0.1',port),make_handler(store,token,f'127.0.0.1:{port}'))
    server.daemon_threads=True
    print(f'Research2 optional lab: http://127.0.0.1:{port}',flush=True)
    print(f'Outputs: {store.root}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        for identifier in list(store.jobs):store.cancel(identifier)
        server.server_close()
