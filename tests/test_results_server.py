import functools
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer
from scripts.serve_results import ResultsHandler


class ResultsServerTests(unittest.TestCase):
    def test_mp4_byte_ranges_for_browser_seeking(self):
        with tempfile.TemporaryDirectory() as temp:
            payload=bytes(range(100));(Path(temp)/'test.mp4').write_bytes(payload)
            server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(ResultsHandler,directory=temp))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}/test.mp4'
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=10-19'})) as r:
                    self.assertEqual(r.status,206);self.assertEqual(r.headers['Content-Range'],'bytes 10-19/100')
                    self.assertEqual(r.read(),payload[10:20]);self.assertEqual(r.headers['Content-Type'],'video/mp4')
                with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=-4'})) as r:
                    self.assertEqual(r.read(),payload[-4:])
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=200-'}))
                self.assertEqual(error.exception.code,416)
            finally:server.shutdown();server.server_close();thread.join()
