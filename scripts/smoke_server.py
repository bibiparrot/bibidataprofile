"""Verify the packaged marimo app serves HTTP, then cleanly stop the test server."""
from pathlib import Path
import os
import subprocess
import sys
import time
import urllib.request

root = Path(__file__).resolve().parents[1]
log_path = root / ".cache" / "server-smoke.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
environment = dict(os.environ)
environment["XDG_CONFIG_HOME"] = str(root / ".cache" / "config")
environment["PYTHONPATH"] = str(root / "src")
with log_path.open("w", encoding="utf-8") as log:
    process = subprocess.Popen([
        sys.executable, "-m", "marimo", "run",
        str(root / "src/bibidataprofile/marimo_app.py"),
        "--host", "127.0.0.1", "--port", "2739", "--headless", "--no-token",
    ], env=environment, stdout=log, stderr=log,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError(log_path.read_text(encoding="utf-8"))
            try:
                with urllib.request.urlopen("http://127.0.0.1:2739", timeout=1) as response:
                    assert response.status == 200
                    assert b"marimo" in response.read().lower()
                print("marimo server HTTP smoke test passed")
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Server did not become ready: " + log_path.read_text(encoding="utf-8"))
    finally:
        process.terminate()
        process.wait(timeout=15)
