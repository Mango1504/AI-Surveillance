"""
AI Surveillance System Launcher
Starts the backend server and React frontend dashboard.
Compiled to .exe using PyInstaller for one-click launch.
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading
import ctypes
from pathlib import Path


def get_app_dir():
    """Get the application root directory."""
    if getattr(sys, 'frozen', False):
        # Running as compiled .exe
        return Path(sys.executable).parent
    else:
        # Running as script
        return Path(__file__).parent


def set_console_title(title):
    """Set the console window title."""
    try:
        ctypes.windll.kernel32.SetConsoleTitleW(title)
    except Exception:
        pass


def find_python():
    """Find the Python executable (venv or system)."""
    app_dir = get_app_dir()
    venv_python = app_dir / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)

    # Try conda env
    conda_python = app_dir / ".venv" / "python.exe"
    if conda_python.exists():
        return str(conda_python)

    # Fall back to system Python
    return "python"


def find_npm():
    """Find npm executable."""
    # Check common locations
    npm_paths = [
        r"C:\Program Files\nodejs\npm.cmd",
        r"C:\Program Files (x86)\nodejs\npm.cmd",
    ]
    for p in npm_paths:
        if os.path.exists(p):
            return p

    # Try PATH
    return "npm"


def start_backend(app_dir, python_exe):
    """Start the Flask backend server."""
    backend_dir = app_dir / "surveillance-app" / "backend"
    main_script = backend_dir / "main_proctor.py"

    if not main_script.exists():
        # Try alternative main files
        alt_scripts = ["main.py", "app.py"]
        for alt in alt_scripts:
            alt_path = backend_dir / alt
            if alt_path.exists():
                main_script = alt_path
                break

    if not main_script.exists():
        print(f"  [ERROR] Backend script not found in {backend_dir}")
        return None

    print(f"  Starting: {main_script.name}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(backend_dir)

    process = subprocess.Popen(
        [python_exe, "-u", str(main_script)],
        cwd=str(backend_dir),
        env=env,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    return process


def start_frontend(app_dir):
    """Start the React development server."""
    frontend_dir = app_dir / "surveillance-app"
    package_json = frontend_dir / "package.json"

    if not package_json.exists():
        print(f"  [ERROR] package.json not found in {frontend_dir}")
        return None

    npm_exe = find_npm()
    print(f"  Starting: npm run dev")

    process = subprocess.Popen(
        [npm_exe, "run", "dev"],
        cwd=str(frontend_dir),
        shell=True,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    return process


def open_browser_delayed(url, delay=8):
    """Open the browser after a delay to let servers start."""
    time.sleep(delay)
    print(f"\n  Opening browser: {url}")
    webbrowser.open(url)


def print_banner():
    """Print the application banner."""
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║         AI SURVEILLANCE SYSTEM                   ║")
    print("  ║         NVIDIA Metropolis Integration            ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()


def print_status(backend_ok, frontend_ok):
    """Print service status."""
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    b_icon = "✓" if backend_ok else "✗"
    f_icon = "✓" if frontend_ok else "✗"
    print(f"  │  [{b_icon}] Backend Server    → http://localhost:5000  │")
    print(f"  │  [{f_icon}] React Dashboard   → http://localhost:3000  │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print("  AI models are loading (~10 seconds)...")
    print("  Dashboard will open automatically in your browser.")
    print()
    print("  Press Ctrl+C or close this window to stop all services.")
    print()


def main():
    set_console_title("AI Surveillance System")
    print_banner()

    app_dir = get_app_dir()
    print(f"  App directory: {app_dir}")
    print()

    # Find Python
    python_exe = find_python()
    print(f"  Python: {python_exe}")

    processes = []

    # Start Backend
    print()
    print("  [1/2] Starting Backend Server...")
    backend_proc = start_backend(app_dir, python_exe)
    if backend_proc:
        processes.append(backend_proc)
        print("        Backend started (PID: {})".format(backend_proc.pid))
    else:
        print("        Backend failed to start!")

    # Start Frontend
    print()
    print("  [2/2] Starting React Dashboard...")
    frontend_proc = start_frontend(app_dir)
    if frontend_proc:
        processes.append(frontend_proc)
        print("        Frontend started (PID: {})".format(frontend_proc.pid))
    else:
        print("        Frontend failed to start!")

    # Print status
    print_status(backend_proc is not None, frontend_proc is not None)

    # Open browser after delay
    browser_thread = threading.Thread(
        target=open_browser_delayed,
        args=("http://localhost:3000", 8),
        daemon=True,
    )
    browser_thread.start()

    # Wait for user to close
    try:
        while True:
            # Check if processes are still running
            all_dead = all(p.poll() is not None for p in processes)
            if all_dead and processes:
                print("\n  All services have stopped.")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\n  Shutting down...")

    # Cleanup
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    print("  All services stopped. Goodbye!")
    time.sleep(2)


if __name__ == "__main__":
    main()
