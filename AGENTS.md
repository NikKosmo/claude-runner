# claude-runner — Technical Guide

**Multi-project workspace:** This project is part of a larger workspace. If `../AGENTS.md` exists, **you MUST read it before proceeding** — it contains cross-project standards and shared rules (`common_rules/`).

## Purpose

Clean subprocess wrapper for `claude -p` CLI invocations. Handles env isolation, hooks bypass, markdown fence stripping, and JSON extraction so callers don't have to.

## Quick Commands

```bash
# Run tests
.venv/bin/python -m pytest tests/ -q --tb=short

# Lint
.venv/bin/ruff check .
.venv/bin/ruff format --check .

# Type check
.venv/bin/pyright src/

# Install for development
python3 -m venv .venv
.venv/bin/pip install -r requirements_test.txt
.venv/bin/pip install -e .
```

## Public API

```python
from claude_runner import run_claude, run_claude_json

# Raw text response
text = run_claude("summarize this", model="claude-sonnet-4-6", timeout=90)

# Parsed JSON response
data = run_claude_json("extract as JSON: {shop, date, total}", timeout=60)
```

**Exceptions:** `ClaudeError`, `ClaudeTimeoutError`, `JsonParseError`

## Architecture

Single module: `src/claude_runner/runner.py`

- `run_claude()` → `subprocess.run` with env/cwd isolation → raw text
- `run_claude_json()` → `run_claude()` → fence strip → `json.JSONDecoder.raw_decode` → dict
- Env isolation: strips `CLAUDECODE`, uses `--setting-sources local`, cwd `~/.config/nohooks`
- Zero runtime dependencies (stdlib only)
