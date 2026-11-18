#!/usr/bin/env python3
"""Serve a results folder locally, including byte-range MP4 playback and seeking."""
import argparse
import functools
import mimetypes
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil


class ResultsHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self._byte_range = None
        path = Path(self.translate_path(self.path))
        value = self.headers.get('Range')
        if not value or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', value.strip())
        if not match or not any(match.groups()):
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
        size = path.stat().st_size
        if match.group(1):
            start = int(match.group(1));end = min(size-1,int(match.group(2))) if match.group(2) else size-1
        else:
            suffix = int(match.group(2));start = max(0,size-suffix);end = size-1
        if start >= size or start > end:
            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers()
            return None
        source = path.open('rb');source.seek(start)
        self.send_response(206)
        self.send_header('Content-Type',self.guess_type(str(path)))
        self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length',str(end-start+1))
        self.send_header('Accept-Ranges','bytes')
        self._byte_range=(start,end);self.end_headers()
        return source

    def end_headers(self):
        if self._byte_range is None:self.send_header('Accept-Ranges','bytes')
        super().end_headers()

    def copyfile(self,source,outputfile):
        if self._byte_range is None:
            return super().copyfile(source,outputfile)
        remaining=self._byte_range[1]-self._byte_range[0]+1
        while remaining:
            block=source.read(min(1024*1024,remaining))
            if not block:break
            try:outputfile.write(block)
            except (BrokenPipeError,ConnectionResetError):return
            remaining-=len(block)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--port',type=int,default=18764);parser.add_argument('--open',action='store_true',help='Open the atlas in the default browser');args=parser.parse_args()
    root=args.directory.expanduser().resolve()
    if not root.is_dir():parser.error('Existing results directory required')
    handler=functools.partial(ResultsHandler,directory=str(root))
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler)
    url=f'http://127.0.0.1:{server.server_port}/START_HERE.html'
    print(f'Open {url}',flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
