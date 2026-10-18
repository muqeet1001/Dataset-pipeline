"""Serve the local viewer and run files on loopback only."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=9086)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    handler=partial(SimpleHTTPRequestHandler,directory=str(root))
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler)
    print(f'Clothing viewer: http://127.0.0.1:{args.port}/web/clothing-viewer/',flush=True)
    server.serve_forever()
