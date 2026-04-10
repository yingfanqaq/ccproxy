# proxy

Local fork of the current `ccproxy` runtime and package, isolated from future `uv tool` updates.

## Layout

- `ccproxy/`: active source tree copied from the currently working installation
- `runtime/`: local copied runtime/venv used by the launcher, intentionally gitignored
- `bin/ccproxy-local`: launcher that runs local source with the local runtime
- `scripts/claude_source.py`: switch Claude Code between `codex`, `gemini`, and native Anthropic

## Common commands

```bash
~/mycodelibrary/proxy/bin/ccproxy-local serve --host 127.0.0.1 --port 18112 --auth-token ccproxy-local-token
python3 ~/mycodelibrary/proxy/scripts/claude_source.py codex
python3 ~/mycodelibrary/proxy/scripts/claude_source.py gemini
python3 ~/mycodelibrary/proxy/scripts/claude_source.py native
python3 ~/mycodelibrary/proxy/scripts/claude_source.py show
```
