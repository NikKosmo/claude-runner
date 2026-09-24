"""Core claude -p subprocess runner."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_NOHOOKS_DIR = Path.home() / ".config" / "nohooks"
_KEEP_QUARANTINE_ENV = "CLAUDE_RUNNER_KEEP_QUARANTINE"
_quarantine_seen: set[str] = set()

_log = logging.getLogger(__name__)


class ClaudeError(RuntimeError):
    """Claude CLI invocation failed."""


class ClaudeTimeoutError(ClaudeError):
    """Claude CLI timed out."""


class JsonParseError(ClaudeError):
    """Claude returned text that could not be parsed as JSON."""

    def __init__(self, raw_output: str, cause: Exception) -> None:
        self.raw_output = raw_output
        self.cause = cause
        super().__init__(f"Failed to parse JSON from Claude output: {cause}")


def run_claude(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int = 60,
    add_dirs: list[str | Path] | None = None,
) -> str:
    """Run ``claude -p`` and return the raw text response.

    Handles env isolation (``CLAUDECODE`` removal), hooks bypass
    (``--setting-sources local``, nohooks cwd), and error wrapping.

    Args:
        add_dirs: Directories to grant Claude read access to (``--add-dir``).
            Use when the prompt references files Claude needs to read.

    Raises:
        ClaudeTimeoutError: if the CLI does not respond within *timeout* seconds.
        ClaudeError: on non-zero exit or OS-level failure.
    """
    _clear_gatekeeper_quarantine()
    cmd, stdin_input = _build_command(prompt, model=model, add_dirs=add_dirs)
    env = _clean_env()
    cwd = _ensure_nohooks_dir()

    try:
        result = subprocess.run(
            cmd,
            input=stdin_input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClaudeTimeoutError(f"claude -p timed out after {timeout}s") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise ClaudeError(f"claude -p failed (exit {exc.returncode}): {stderr}") from exc
    except FileNotFoundError as exc:
        raise ClaudeError("claude CLI not found on PATH") from exc
    except OSError as exc:
        raise ClaudeError(f"Failed to run claude CLI: {exc}") from exc

    return result.stdout.strip()


def run_claude_json(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int = 60,
    add_dirs: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Run ``claude -p`` and return the response parsed as JSON.

    Strips markdown fences and extracts the JSON object automatically.
    The caller receives a dict — never raw text.

    Args:
        add_dirs: Directories to grant Claude read access to (``--add-dir``).

    Raises:
        JsonParseError: if the response cannot be parsed as JSON.
        ClaudeTimeoutError: if the CLI does not respond within *timeout* seconds.
        ClaudeError: on non-zero exit or OS-level failure.
    """
    raw = run_claude(prompt, model=model, timeout=timeout, add_dirs=add_dirs)
    return _parse_json(raw)


def _clear_gatekeeper_quarantine() -> None:
    """Remove com.apple.quarantine from the claude binary when macOS re-applies it.

    The CLI is distributed as a Homebrew cask, so every upgrade installs a fresh
    binary carrying a fresh quarantine stamp. A human clicks through the resulting
    Gatekeeper dialog once. A background job cannot: the child process waits on a
    dialog nobody will ever answer, and the call dies at its timeout. Observed
    2026-09-24 — four generations in a row raising
    ``claude -p timed out after 120s`` the morning after an upgrade, with a healthy
    binary and a valid login.

    Scope is one file: the resolved claude executable, checked once per process.
    Removing the attribute needs no privileges. Set
    ``CLAUDE_RUNNER_KEEP_QUARANTINE=1`` to switch this off and keep Gatekeeper's
    first-run prompt, at the cost of that hang in unattended use.

    Best-effort by design: any failure is logged and ignored, because the run that
    follows is a better error message than anything raised from here.
    """
    if sys.platform != "darwin" or os.environ.get(_KEEP_QUARANTINE_ENV):
        return
    resolved = shutil.which("claude")
    if resolved is None:
        return
    real = os.path.realpath(resolved)
    if real in _quarantine_seen:
        return
    _quarantine_seen.add(real)
    try:
        listed = subprocess.run(["xattr", real], capture_output=True, text=True, timeout=5)
        if "com.apple.quarantine" not in listed.stdout:
            return
        subprocess.run(
            ["xattr", "-d", "com.apple.quarantine", real],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        _log.info("Cleared com.apple.quarantine from %s", real)
    except (OSError, subprocess.SubprocessError) as exc:
        _log.warning("Could not clear com.apple.quarantine from %s: %s", real, exc)


def _build_command(
    prompt: str, *, model: str | None, add_dirs: list[str | Path] | None = None
) -> tuple[list[str], str | None]:
    """Build CLI command. Returns (cmd, stdin_input).

    When --add-dir is used, the prompt goes via stdin because --add-dir
    is variadic and swallows the positional prompt argument.
    """
    cmd = ["claude", "-p", "--setting-sources", "local"]
    if model is not None:
        cmd.extend(["--model", model])
    if add_dirs:
        for d in add_dirs:
            cmd.extend(["--add-dir", str(d)])
        return cmd, prompt
    cmd.append(prompt)
    return cmd, None


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}


def _ensure_nohooks_dir() -> Path:
    """Ensure the nohooks directory exists, creating it if needed."""
    _NOHOOKS_DIR.mkdir(parents=True, exist_ok=True)
    return _NOHOOKS_DIR


def _strip_fences(text: str) -> str:
    """Remove markdown code fences wrapping.

    Only strips fences when the text starts with ``` and the closing ```
    is on its own line (standard markdown fence format).
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    if len(lines) < 3:
        return stripped
    # Check that last line is a closing fence
    if not lines[-1].strip().startswith("```"):
        return stripped
    # Drop first and last lines (opening and closing fences)
    return "\n".join(lines[1:-1]).strip()


def _parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from Claude output, handling fences and preamble."""
    cleaned = _strip_fences(text)

    # Try direct parse first
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        raise JsonParseError(text, TypeError(f"Expected dict, got {type(result).__name__}"))
    except json.JSONDecodeError:
        pass

    # Fallback: find valid JSON by trying each '{' with raw_decode,
    # which stops at the end of the JSON object and ignores trailing text.
    decoder = json.JSONDecoder()
    pos = 0
    last_error: Exception | None = None
    while True:
        start = cleaned.find("{", pos)
        if start == -1:
            break
        try:
            result, _ = decoder.raw_decode(cleaned, start)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError as exc:
            last_error = exc
        pos = start + 1

    raise JsonParseError(text, last_error or ValueError("No JSON object found in response"))
