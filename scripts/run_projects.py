import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def run_checked(command):
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def is_http_alive(url):
    try:
        with urlopen(url, timeout=0.5):
            return True
    except HTTPError:
        return True
    except URLError:
        return False
    except TimeoutError:
        return False


def start_process(name, command, health_url):
    if is_http_alive(health_url):
        print(f"Using already-running {name}.", flush=True)
        return None
    print(f"Starting {name}...", flush=True)
    return subprocess.Popen(command, cwd=ROOT)


def main():
    print("Training offline ML models and refreshing report artifacts.", flush=True)
    run_checked([sys.executable, "-m", "crisislens.ml.train"])
    run_checked([sys.executable, "-m", "leafguard.ml.train"])

    services = [
        ("CrisisLens API", [sys.executable, "-m", "crisislens.api.server"], "http://127.0.0.1:8011/health"),
        ("LeafGuard API", [sys.executable, "-m", "leafguard.api.server"], "http://127.0.0.1:8022/health"),
        (
            "CrisisLens UI",
            ["npm", "--prefix", "crisislens/ui", "run", "dev", "--", "--port", "5173"],
            "http://127.0.0.1:5173",
        ),
        (
            "LeafGuard UI",
            ["npm", "--prefix", "leafguard/ui", "run", "dev", "--", "--port", "5174"],
            "http://127.0.0.1:5174",
        ),
    ]
    processes = [start_process(name, command, health_url) for name, command, health_url in services]

    def shutdown(_signum=None, _frame=None):
        print("\nStopping project services...", flush=True)
        for process in processes:
            if process is not None and process.poll() is None:
                process.terminate()
        for process in processes:
            if process is None:
                continue
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        print("Stopped.", flush=True)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("", flush=True)
    print("Projects are running:", flush=True)
    print("  CrisisLens API: http://127.0.0.1:8011", flush=True)
    print("  LeafGuard API:  http://127.0.0.1:8022", flush=True)
    print("  CrisisLens UI:  http://127.0.0.1:5173", flush=True)
    print("  LeafGuard UI:   http://127.0.0.1:5174", flush=True)
    print("", flush=True)
    print("Press Ctrl+C to stop APIs and UIs.", flush=True)

    while True:
        for name, process in zip([name for name, _, _ in services], processes):
            if process is None:
                continue
            code = process.poll()
            if code is not None:
                print(f"{name} exited with code {code}.", flush=True)
                shutdown()
        time.sleep(1)


if __name__ == "__main__":
    main()
