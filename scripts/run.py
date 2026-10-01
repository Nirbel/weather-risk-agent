"""Start the FastAPI backend and the Streamlit UI together. Ctrl-C stops both.

    uv run python scripts/run.py                 # API http://localhost:8000, UI http://localhost:8501
    uv run python scripts/run.py --host 0.0.0.0  # inside Docker

If either process exits, the other is stopped too, so you never end up with half an app.
"""

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pump(proc: subprocess.Popen, prefix: str) -> None:
    for line in proc.stdout:
        print(f"[{prefix}] {line}", end="", flush=True)


def port_free(host: str, port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1" if host == "0.0.0.0" else host, port)) != 0


def wait_for_api(url: str, proc: subprocess.Popen, timeout: float = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    args = parser.parse_args()

    busy = [p for p in (args.api_port, args.ui_port) if not port_free(args.host, p)]
    if busy:
        print(f"[run] Port(s) {', '.join(map(str, busy))} already in use — is the app already running? "
              "Stop it or pass --api-port / --ui-port.", flush=True)
        return 1
    api_url = f"http://127.0.0.1:{args.api_port}"
    env = os.environ | {"API_URL": os.environ.get("API_URL", api_url), "PYTHONUNBUFFERED": "1"}
    common = {"cwd": ROOT, "env": env, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}

    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "weather_risk.api.main:app",
                            "--host", args.host, "--port", str(args.api_port)], **common)
    threading.Thread(target=pump, args=(api, "api"), daemon=True).start()
    processes = [api]
    try:
        if not wait_for_api(api_url, api):
            print("[run] API did not become healthy — see the [api] log above.", flush=True)
            return 1
        ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py",
                               "--server.address", args.host, "--server.port", str(args.ui_port),
                               "--server.headless", "true", "--browser.gatherUsageStats", "false"], **common)
        threading.Thread(target=pump, args=(ui, "ui"), daemon=True).start()
        processes.append(ui)
        print(f"[run] API  {api_url}/docs\n[run] UI   http://localhost:{args.ui_port}\n[run] Ctrl-C to stop.", flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(0.5)
        print("[run] A process exited — stopping the other.", flush=True)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    sys.exit(main())
