"""Core claude -p subprocess runner."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

_NOHOOKS_DIR = Path.home() / ".config" / "nohooks"


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
) -> str:
    """Run ``claude -p`` and return the raw text response.

    Handles env isolation (``CLAUDECODE`` removal), hooks bypass
    (``--setting-sources local``, nohooks cwd), and error wrapping.

    Raises:
        ClaudeTimeoutError: if the CLI does not respond within *timeout* seconds.
        ClaudeError: on non-zero exit or OS-level failure.
    """
    cmd = _build_command(prompt, model=model)
    env = _clean_env()
    cwd = _ensure_nohooks_dir()

    try:
        result = subprocess.run(
            cmd,
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
) -> dict[str, Any]:
    """Run ``claude -p`` and return the response parsed as JSON.

    Strips markdown fences and extracts the JSON object automatically.
    The caller receives a dict — never raw text.

    Raises:
        JsonParseError: if the response cannot be parsed as JSON.
        ClaudeTimeoutError: if the CLI does not respond within *timeout* seconds.
        ClaudeError: on non-zero exit or OS-level failure.
    """
    raw = run_claude(prompt, model=model, timeout=timeout)
    return _parse_json(raw)


def _build_command(prompt: str, *, model: str | None) -> list[str]:
    cmd = ["claude", "-p", "--setting-sources", "local"]
    if model is not None:
        cmd.extend(["--model", model])
    cmd.append(prompt)
    return cmd


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
