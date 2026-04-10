#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PYTHON = PROJECT_ROOT / "runtime" / "bin" / "python"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "ccproxy" / "local.json"
DEFAULT_STATE_DIR = Path.home() / ".local" / "state" / "ccproxy"
DEFAULT_CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
GENERIC_LAUNCH_AGENT_LABEL = "com.ccproxy.local"
LEGACY_LAUNCH_AGENT_LABEL = "com.yingfanqaq.ccproxy"


def run_subprocess(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
        env=env,
    )


def _extract_flag(arguments: list[str], flag: str) -> str | None:
    for index, value in enumerate(arguments):
        if value == flag and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return None


def _load_plist(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            data = plistlib.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _infer_existing_launch_agent() -> dict[str, Any]:
    launch_dir = Path.home() / "Library" / "LaunchAgents"
    if not launch_dir.exists():
        return {}

    candidates = [
        launch_dir / f"{GENERIC_LAUNCH_AGENT_LABEL}.plist",
        launch_dir / f"{LEGACY_LAUNCH_AGENT_LABEL}.plist",
    ]
    candidates.extend(sorted(launch_dir.glob("*ccproxy*.plist")))

    for path in candidates:
        if not path.exists():
            continue
        data = _load_plist(path)
        if not data:
            continue
        program_args = data.get("ProgramArguments") or []
        joined = " ".join(str(item) for item in program_args)
        if "ccproxy" not in joined:
            continue

        env = data.get("EnvironmentVariables") or {}
        inferred: dict[str, Any] = {
            "launch_agent_label": data.get("Label") or path.stem,
            "launch_agent_path": str(path),
            "stdout_log_path": data.get("StandardOutPath"),
            "stderr_log_path": data.get("StandardErrorPath"),
            "host": _extract_flag(program_args, "--host"),
            "port": _extract_flag(program_args, "--port"),
            "auth_token": _extract_flag(program_args, "--auth-token"),
            "upstream_proxy_url": env.get("HTTPS_PROXY")
            or env.get("HTTP_PROXY")
            or env.get("https_proxy")
            or env.get("http_proxy"),
        }
        return {key: value for key, value in inferred.items() if value not in (None, "")}
    return {}


def _default_settings() -> dict[str, Any]:
    generic_agent_path = (
        Path.home() / "Library" / "LaunchAgents" / f"{GENERIC_LAUNCH_AGENT_LABEL}.plist"
    )
    defaults: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 18112,
        "auth_token": "ccproxy-local-token",
        "upstream_proxy_url": "http://127.0.0.1:7897",
        "launch_agent_label": GENERIC_LAUNCH_AGENT_LABEL,
        "launch_agent_path": str(generic_agent_path),
        "stdout_log_path": str(DEFAULT_STATE_DIR / "launchd.out.log"),
        "stderr_log_path": str(DEFAULT_STATE_DIR / "launchd.err.log"),
        "claude_settings_path": str(DEFAULT_CLAUDE_SETTINGS),
    }

    inferred = _infer_existing_launch_agent()
    defaults.update(inferred)
    if isinstance(defaults.get("port"), str) and str(defaults["port"]).isdigit():
        defaults["port"] = int(defaults["port"])
    return defaults


def _resolve_config_path(config_path: str | None) -> Path:
    return Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH


def load_local_config(config_path: str | None = None) -> tuple[Path, dict[str, Any]]:
    path = _resolve_config_path(config_path)
    config = _default_settings()
    if path.exists():
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            config.update(data)
    if isinstance(config.get("port"), str) and str(config["port"]).isdigit():
        config["port"] = int(config["port"])
    return path, config


def save_local_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def apply_global_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    for key in (
        "host",
        "port",
        "auth_token",
        "upstream_proxy_url",
        "launch_agent_label",
        "launch_agent_path",
        "stdout_log_path",
        "stderr_log_path",
        "claude_settings_path",
    ):
        value = getattr(args, key, None)
        if value is not None:
            updated[key] = value
    return updated


def build_proxy_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    proxy_url = config.get("upstream_proxy_url")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if proxy_url:
            env[key] = str(proxy_url)
        else:
            env.pop(key, None)
    for key in ("ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    env["NO_PROXY"] = env.get("NO_PROXY", "127.0.0.1,localhost")
    env["no_proxy"] = env.get("no_proxy", env["NO_PROXY"])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else f"{PROJECT_ROOT}:{existing_pythonpath}"
    )
    return env


def print_config(config: dict[str, Any]) -> None:
    for key in (
        "host",
        "port",
        "auth_token",
        "upstream_proxy_url",
        "launch_agent_label",
        "launch_agent_path",
        "stdout_log_path",
        "stderr_log_path",
        "claude_settings_path",
    ):
        print(f"{key} = {config.get(key)}")


def read_claude_settings(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(config["claude_settings_path"])).expanduser()
    if not path.exists():
        return {"env": {}, "model": None}
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {"env": {}, "model": None}


def write_claude_settings(config: dict[str, Any], data: dict[str, Any]) -> None:
    path = Path(str(config["claude_settings_path"])).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def show_source(config: dict[str, Any]) -> None:
    data = read_claude_settings(config)
    env = data.get("env", {}) if isinstance(data.get("env"), dict) else {}
    print("ANTHROPIC_BASE_URL =", env.get("ANTHROPIC_BASE_URL"))
    print("ANTHROPIC_AUTH_TOKEN =", env.get("ANTHROPIC_AUTH_TOKEN"))
    print("model =", data.get("model"))


def use_source(config: dict[str, Any], mode: str) -> None:
    data = read_claude_settings(config)
    env = data.setdefault("env", {})
    if not isinstance(env, dict):
        env = {}
        data["env"] = env

    if mode == "native":
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        write_claude_settings(config, data)
        print("Claude source -> native Anthropic API")
        print(
            "Note: this requires your real ANTHROPIC_API_KEY to be available separately."
        )
        show_source(config)
        return

    env["ANTHROPIC_BASE_URL"] = f"http://{config['host']}:{config['port']}/{mode}"
    env["ANTHROPIC_AUTH_TOKEN"] = str(config["auth_token"])
    write_claude_settings(config, data)
    print(f"Claude source -> {mode} @ http://{config['host']}:{config['port']}/{mode}")
    show_source(config)


def build_launch_agent_payload(config: dict[str, Any]) -> dict[str, Any]:
    stdout_path = Path(str(config["stdout_log_path"])).expanduser()
    stderr_path = Path(str(config["stderr_log_path"])).expanduser()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    env_vars: dict[str, str] = {}
    proxy_url = config.get("upstream_proxy_url")
    if proxy_url:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env_vars[key] = str(proxy_url)

    return {
        "Label": str(config["launch_agent_label"]),
        "ProgramArguments": [
            str(PROJECT_ROOT / "bin" / "ccproxy"),
            "serve",
            "--host",
            str(config["host"]),
            "--port",
            str(config["port"]),
            "--auth-token",
            str(config["auth_token"]),
        ],
        "WorkingDirectory": str(PROJECT_ROOT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        "EnvironmentVariables": env_vars,
    }


def write_launch_agent(config: dict[str, Any]) -> Path:
    path = Path(str(config["launch_agent_path"])).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_launch_agent_payload(config)
    with path.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=False)
    return path


def launchctl_target(config: dict[str, Any]) -> str:
    return f"gui/{os.getuid()}/{config['launch_agent_label']}"


def health_status(config: dict[str, Any]) -> tuple[bool, str]:
    url = f"http://{config['host']}:{config['port']}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with opener.open(request, timeout=3) as response:
            body = response.read().decode("utf-8", errors="replace")
        return True, body
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def service_install(config: dict[str, Any]) -> None:
    path = write_launch_agent(config)
    print(f"LaunchAgent written to {path}")
    print(f"label = {config['launch_agent_label']}")
    print(f"stdout_log = {config['stdout_log_path']}")
    print(f"stderr_log = {config['stderr_log_path']}")


def service_start(config: dict[str, Any]) -> None:
    service_install(config)
    path = Path(str(config["launch_agent_path"])).expanduser()
    target = launchctl_target(config)
    run_subprocess(["launchctl", "bootout", target], check=False)
    run_subprocess(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)])
    print(f"started {target}")


def service_stop(config: dict[str, Any]) -> None:
    target = launchctl_target(config)
    run_subprocess(["launchctl", "bootout", target], check=False)
    print(f"stopped {target}")


def service_restart(config: dict[str, Any]) -> None:
    service_install(config)
    target = launchctl_target(config)
    path = Path(str(config["launch_agent_path"])).expanduser()
    run_subprocess(["launchctl", "bootout", target], check=False)
    run_subprocess(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)])
    print(f"restarted {target}")


def service_status(config: dict[str, Any]) -> int:
    target = launchctl_target(config)
    result = run_subprocess(["launchctl", "print", target], check=False, capture=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    ok, detail = health_status(config)
    print("---")
    print(f"health_ok = {ok}")
    print(f"health = {detail}")
    return result.returncode


def service_logs(config: dict[str, Any], *, follow: bool, lines: int, stream: str) -> int:
    selected: list[Path] = []
    if stream in {"stdout", "both"}:
        selected.append(Path(str(config["stdout_log_path"])).expanduser())
    if stream in {"stderr", "both"}:
        selected.append(Path(str(config["stderr_log_path"])).expanduser())

    if not selected:
        print("No log stream selected.", file=sys.stderr)
        return 1

    if follow and len(selected) == 1:
        return run_subprocess(
            ["tail", "-n", str(lines), "-f", str(selected[0])],
            check=False,
        ).returncode

    for index, path in enumerate(selected):
        if index:
            print("---")
        print(f"==> {path} <==")
        if path.exists():
            run_subprocess(["tail", "-n", str(lines), str(path)], check=False)
        else:
            print("(missing)")
    if follow and len(selected) > 1:
        print(
            "follow with multiple streams is not supported in a single process; rerun with --stdout or --stderr",
            file=sys.stderr,
        )
    return 0


def run_upstream(arguments: list[str], config: dict[str, Any]) -> int:
    env = build_proxy_env(config)
    command = [str(RUNTIME_PYTHON), "-m", "ccproxy"]
    if arguments and arguments[0] == "serve":
        serve_args = list(arguments[1:])
        if _extract_flag(serve_args, "--host") is None:
            serve_args = ["--host", str(config["host"])] + serve_args
        if _extract_flag(serve_args, "--port") is None:
            serve_args = ["--port", str(config["port"])] + serve_args
        if _extract_flag(serve_args, "--auth-token") is None:
            serve_args = ["--auth-token", str(config["auth_token"])] + serve_args
        command.extend(["serve", *serve_args])
    else:
        command.extend(arguments)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return completed.returncode


def print_doctor(config_path: Path, config: dict[str, Any]) -> int:
    print(f"project_root = {PROJECT_ROOT}")
    print(f"runtime_python = {RUNTIME_PYTHON}")
    print(f"config_path = {config_path}")
    print_config(config)
    print("---")
    show_source(config)
    print("---")
    service_status(config)
    return 0


def maybe_persist_settings(
    command: str,
    config_path: Path,
    base: dict[str, Any],
    current: dict[str, Any],
) -> None:
    if command in {"service", "settings"} and base != current:
        save_local_config(config_path, current)


def build_global_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-path")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--auth-token")
    parser.add_argument("--upstream-proxy-url")
    parser.add_argument("--launch-agent-label")
    parser.add_argument("--launch-agent-path")
    parser.add_argument("--stdout-log-path")
    parser.add_argument("--stderr-log-path")
    parser.add_argument("--claude-settings-path")
    return parser


def handle_source(arguments: list[str], config: dict[str, Any]) -> int:
    parser = argparse.ArgumentParser(prog="ccproxy source")
    sub = parser.add_subparsers(dest="action", required=True)
    use_parser = sub.add_parser("use")
    use_parser.add_argument("mode", choices=["codex", "gemini", "native"])
    sub.add_parser("show")
    args = parser.parse_args(arguments)
    if args.action == "show":
        show_source(config)
        return 0
    use_source(config, args.mode)
    return 0


def handle_service(arguments: list[str], config: dict[str, Any]) -> int:
    parser = argparse.ArgumentParser(prog="ccproxy service")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("install")
    sub.add_parser("start")
    sub.add_parser("stop")
    sub.add_parser("restart")
    sub.add_parser("status")
    logs_parser = sub.add_parser("logs")
    logs_parser.add_argument("-f", "--follow", action="store_true")
    logs_parser.add_argument("-n", "--lines", type=int, default=80)
    stream = logs_parser.add_mutually_exclusive_group()
    stream.add_argument("--stdout", action="store_true")
    stream.add_argument("--stderr", action="store_true")
    args = parser.parse_args(arguments)

    if args.action == "install":
        service_install(config)
        return 0
    if args.action == "start":
        service_start(config)
        return 0
    if args.action == "stop":
        service_stop(config)
        return 0
    if args.action == "restart":
        service_restart(config)
        return 0
    if args.action == "status":
        return service_status(config)

    stream_name = "both"
    if args.stdout:
        stream_name = "stdout"
    elif args.stderr:
        stream_name = "stderr"
    return service_logs(config, follow=args.follow, lines=args.lines, stream=stream_name)


def handle_settings(arguments: list[str], config_path: Path, config: dict[str, Any]) -> int:
    parser = argparse.ArgumentParser(prog="ccproxy settings")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("show")
    set_parser = sub.add_parser("set")
    set_parser.add_argument("--host")
    set_parser.add_argument("--port", type=int)
    set_parser.add_argument("--auth-token")
    set_parser.add_argument("--upstream-proxy-url")
    set_parser.add_argument("--launch-agent-label")
    set_parser.add_argument("--launch-agent-path")
    set_parser.add_argument("--stdout-log-path")
    set_parser.add_argument("--stderr-log-path")
    set_parser.add_argument("--claude-settings-path")
    args = parser.parse_args(arguments)
    if args.action == "show":
        print_config(config)
        return 0

    updated = dict(config)
    for key in (
        "host",
        "port",
        "auth_token",
        "upstream_proxy_url",
        "launch_agent_label",
        "launch_agent_path",
        "stdout_log_path",
        "stderr_log_path",
        "claude_settings_path",
    ):
        value = getattr(args, key, None)
        if value is not None:
            updated[key] = value
    save_local_config(config_path, updated)
    print(f"saved {config_path}")
    print_config(updated)
    return 0


def print_help() -> None:
    print(
        "Usage: ccproxy [global options] <command> [args]\n\n"
        "Local management commands:\n"
        "  ccproxy source use codex|gemini|native\n"
        "  ccproxy source show\n"
        "  ccproxy service install|start|stop|restart|status|logs\n"
        "  ccproxy settings show\n"
        "  ccproxy settings set --port 18112 --upstream-proxy-url http://127.0.0.1:7897\n"
        "  ccproxy doctor\n"
        "  ccproxy serve [upstream ccproxy serve options]\n\n"
        "Any other command is forwarded to the upstream ccproxy CLI.\n"
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    global_parser = build_global_parser()
    global_args, remaining = global_parser.parse_known_args(argv)
    config_path, base_config = load_local_config(global_args.config_path)
    config = apply_global_overrides(base_config, global_args)

    if not remaining:
        print_help()
        return 0

    command = remaining[0]
    command_args = remaining[1:]

    if command in {"-h", "--help", "help"}:
        print_help()
        return 0
    if command == "doctor":
        return print_doctor(config_path, config)
    if command == "source":
        return handle_source(command_args, config)
    if command == "service":
        maybe_persist_settings(command, config_path, base_config, config)
        return handle_service(command_args, config)
    if command == "settings":
        return handle_settings(command_args, config_path, config)
    if command == "serve":
        return run_upstream(remaining, config)
    return run_upstream(remaining, config)


if __name__ == "__main__":
    raise SystemExit(main())
