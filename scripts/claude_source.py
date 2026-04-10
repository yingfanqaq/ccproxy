#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "scripts" / "ccproxy_local.py"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: claude_source.py <codex|gemini|native|show> [--port PORT]", file=sys.stderr)
        return 1

    mode = sys.argv[1]
    extra = sys.argv[2:]
    if mode == "show":
        command = [sys.executable, str(TARGET), "source", "show", *extra]
    else:
        command = [sys.executable, str(TARGET), "source", "use", mode, *extra]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
