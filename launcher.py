"""
AI Surveillance System — Launcher Control Panel
Run this file to open the launcher UI in your browser.
"""
import os
import sys
import json
import time
import signal
import socket
import subprocess
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from collections import deque

LAUNCHER_PORT = 8800
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(PROJECT_ROOT, "surveillance-app")
BACKEND_DIR = os.path.join(APP_DIR, "backend")

# ── Process manager ──────────────────────────
processes = {}          # "backend" / "frontend" -> subprocess.Popen
log_buffers = {         # ring-buffer per service
    "backend": deque(maxlen=300),
    "frontend": deque(maxlen=300),
}
lock = threading.Lock()


def _stream_output(name, proc):
    """Read stdout+stderr line-by-line and push into the ring buffer."""
    try:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            with lock:
                log_buffers[name].append(line)
    except Exception:
        pass


def start_service(name):
    with lock:
        if name in processes and processes[name].poll() is None:
            return False  # already running

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    if name == "backend":
        proc = subprocess.Popen(
            [sys.executable, "-u", "main_proctor.py"],
            cwd=BACKEND_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    elif name == "frontend":
        proc = subprocess.Popen(
            ["npm.cmd" if sys.platform == "win32" else "npm", "run", "dev"],
            cwd=APP_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    else:
        return False

    with lock:
        processes[name] = proc
        log_buffers[name].clear()
        log_buffers[name].append(f"[LAUNCHER] {name} started (PID {proc.pid})")

    threading.Thread(target=_stream_output, args=(name, proc), daemon=True).start()
    return True


def stop_service(name):
    with lock:
        proc = processes.get(name)
    if proc is None or proc.poll() is not None:
        return False
    try:
        if sys.platform == "win32":
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.kill()
    with lock:
        log_buffers[name].append(f"[LAUNCHER] {name} stopped.")
    return True


def port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def get_status():
    def _alive(name):
        with lock:
            p = processes.get(name)
        return p is not None and p.poll() is None
    return {
        "backend":  {"running": _alive("backend"),  "port_open": port_open(5000)},
        "frontend": {"running": _alive("frontend"), "port_open": port_open(3000)},
    }


# ── HTTP handler ─────────────────────────────
class LauncherHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress default access logs

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            html_path = os.path.join(PROJECT_ROOT, "launcher.html")
            with open(html_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/api/status":
            self._json(get_status())
        elif self.path.startswith("/api/logs/"):
            name = self.path.split("/")[-1]
            with lock:
                lines = list(log_buffers.get(name, []))
            self._json({"lines": lines})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/start/backend":
            self._json({"ok": start_service("backend")})
        elif self.path == "/api/start/frontend":
            self._json({"ok": start_service("frontend")})
        elif self.path == "/api/stop/backend":
            self._json({"ok": stop_service("backend")})
        elif self.path == "/api/stop/frontend":
            self._json({"ok": stop_service("frontend")})
        elif self.path == "/api/start/all":
            start_service("backend")
            time.sleep(0.3)
            start_service("frontend")
            self._json({"ok": True})
        elif self.path == "/api/stop/all":
            stop_service("frontend")
            stop_service("backend")
            self._json({"ok": True})
        else:
            self.send_error(404)


# ── Entry point ──────────────────────────────
if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", LAUNCHER_PORT), LauncherHandler)
    url = f"http://127.0.0.1:{LAUNCHER_PORT}"
    print(f"\n  +--------------------------------------------+")
    print(f"  |  AI Surveillance -- Launcher Control       |")
    print(f"  |  {url:<42s} |")
    print(f"  +--------------------------------------------+\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[LAUNCHER] Shutting down...")
        stop_service("frontend")
        stop_service("backend")
        server.server_close()
