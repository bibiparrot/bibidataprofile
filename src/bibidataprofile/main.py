"""Launch the marimo control panel."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="BibiDataProfile marimo control panel")
    parser.add_argument("--port", type=int, default=2718)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--edit", action="store_true", help="Open the notebook editor")
    args = parser.parse_args()
    command = [sys.executable, "-m", "marimo", "edit" if args.edit else "run",
               str(Path(__file__).with_name("marimo_app.py")), "--host", "127.0.0.1",
               "--port", str(args.port)]
    if args.headless:
        command.append("--headless")
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
