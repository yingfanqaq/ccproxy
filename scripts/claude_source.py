#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_PORT = 18112
DEFAULT_TOKEN = "ccproxy-local-token"
SETTINGS_PATH = Path.home() / ".claude" / "settings.json"


def load_settings() -> dict:
    return json.loads(SETTINGS_PATH.read_text())


def save_settings(data: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def show_settings(data: dict) -> None:
    env = data.get("env", {})
    print("ANTHROPIC_BASE_URL =", env.get("ANTHROPIC_BASE_URL"))
    print("ANTHROPIC_AUTH_TOKEN =", env.get("ANTHROPIC_AUTH_TOKEN"))
    print("model =", data.get("model"))


def apply_mode(mode: str, port: int) -> None:
    data = load_settings()
    env = data.setdefault("env", {})

    if mode == "native":
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        save_settings(data)
        print("Claude source -> native Anthropic API")
        print("Note: this requires your real ANTHROPIC_API_KEY to be available separately.")
        return

    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}/{mode}"
    env["ANTHROPIC_AUTH_TOKEN"] = DEFAULT_TOKEN
    save_settings(data)
    print(f"Claude source -> {mode} @ http://127.0.0.1:{port}/{mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Switch Claude Code upstream source.")
    parser.add_argument("mode", choices=["codex", "gemini", "native", "show"])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    if args.mode == "show":
        show_settings(load_settings())
        return 0

    apply_mode(args.mode, args.port)
    show_settings(load_settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
