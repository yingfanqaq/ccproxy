# proxy

Local fork of the current `ccproxy` runtime and package, isolated from future `uv tool` updates. The repo now owns local service management, Claude source switching, and shell entrypoints so you do not have to keep large wrappers in `~/.zshrc`.

## Layout

- `ccproxy/`: active source tree copied from the currently working installation
- `runtime/`: local copied runtime/venv used by the launcher, intentionally gitignored
- `bin/ccproxy`: main local entrypoint; use this as the start point for service management, source switching, logs, doctor, and normal `serve`
- `bin/ccproxy-local`: compatibility shim that forwards to `bin/ccproxy`
- `scripts/ccproxy_local.py`: local management CLI implementation
- `scripts/claude_source.py`: compatibility wrapper around `ccproxy source ...`

## Recommended shell integration

Keep `~/.zshrc` minimal and let the repo own the behavior:

```bash
export CCPROXY_PROJECT_ROOT="$HOME/mycodelibrary/proxy"
unalias ccproxy 2>/dev/null
ccproxy() {
  command "${CCPROXY_PROJECT_ROOT}/bin/ccproxy" "$@"
}
```

## Common commands

```bash
ccproxy doctor
ccproxy source show
ccproxy source use codex
ccproxy source use gemini
ccproxy source use native
ccproxy settings show
ccproxy settings set --port 18112 --upstream-proxy-url http://127.0.0.1:7897
ccproxy service status
ccproxy service restart
ccproxy service logs -f
ccproxy serve
```

## Notes

- `ccproxy service ...` will manage the LaunchAgent plist and log paths for you.
- `ccproxy source use ...` updates `~/.claude/settings.json`; restart Claude Code after switching.
- `ccproxy serve` still forwards to the upstream local `ccproxy` CLI and auto-fills host, port, auth token, and proxy defaults.
