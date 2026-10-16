"""Minimal local webhook receiver for testing: prints every JSON payload it is sent.

    python scripts/webhook_receiver.py [port]      # default port 8080

Uses only the standard library, so it runs in a plain python image.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            text = json.dumps(json.loads(body), indent=2)
        except ValueError:
            text = body.decode(errors="replace")
        print(f"--- webhook received on {self.path} ---\n{text}", flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"received": true}')

    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"webhook receiver is running; POST JSON to /webhook")

    def log_message(self, *args) -> None:  # silence the default per-request access log
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"webhook receiver listening on port {port}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
