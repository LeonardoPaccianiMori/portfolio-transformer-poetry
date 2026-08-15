#!/usr/bin/env python3
"""Exercise static-only demo startup, status, and unavailable generation."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sonnet_demo.server import StaticDemoGenerator, create_demo_handler


def main() -> int:
    handler = create_demo_handler(static_root=ROOT / "demo", generator=StaticDemoGenerator())
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/status", timeout=5) as response:
            status = json.load(response)
        if status["status"] != "static_only":
            raise RuntimeError(f"unexpected status: {status}")
        request = urllib.request.Request(
            f"{base}/api/generate",
            data=json.dumps({"opening_line": "Solo et pensoso"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as error:
            payload = json.load(error)
            if error.code != 503 or "static-only" not in payload["error"]:
                raise
        else:
            raise RuntimeError("static-only generation unexpectedly succeeded")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print("static-demo | OK | startup=status generation=unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
