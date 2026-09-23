"""The dashboard server must stop browsers running a stale copy of the
page's code: after the tick labels were fixed, an open tab kept the old
all-"BUY" logic because nothing told the browser to revalidate."""

import http.server
import threading
import urllib.request

from jevloop import serve


def test_dashboard_files_are_served_no_cache(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html></html>")
    monkeypatch.setattr(serve, "DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(serve, "LOG_DIR", tmp_path)
    httpd = http.server.HTTPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=httpd.handle_request, daemon=True).start()
    try:
        port = httpd.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html") as resp:
            assert "no-cache" in resp.headers["Cache-Control"]
    finally:
        httpd.server_close()
