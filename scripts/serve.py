"""Servidor local do painel: http://localhost:8765 (so na propria maquina)."""
import functools
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


handler = functools.partial(NoCache, directory=str(ROOT / "dashboard"))
with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as srv:
    print(f"painel em http://localhost:{PORT}")
    srv.serve_forever()
