"""Tests for claude_runner."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from claude_runner import (
    ClaudeError,
    ClaudeTimeoutError,
    JsonParseError,
    run_claude,
    run_claude_json,
)
from claude_runner.runner import _parse_json, _strip_fences

# --- _strip_fences ---


class TestStripFences:
    def test_no_fences(self):
        assert _strip_fences('{"a": 1}') == '{"a": 1}'

    def test_json_fences(self):
        text = '```json\n{"a": 1}\n```'
        assert _strip_fences(text) == '{"a": 1}'

    def test_plain_fences(self):
        text = '```\n{"a": 1}\n```'
        assert _strip_fences(text) == '{"a": 1}'

    def test_fences_with_trailing_whitespace(self):
        text = '```json\n{"a": 1}\n```\n\n'
        assert _strip_fences(text) == '{"a": 1}'

    def test_no_newline_after_fence_returns_as_is(self):
        text = "```something"
        assert _strip_fences(text) == "```something"

    def test_only_two_lines_returns_as_is(self):
        text = "```json\n```"
        assert _strip_fences(text) == "```json\n```"

    def test_closing_fence_not_on_own_line_returns_as_is(self):
        text = '```json\n{"text": "has ``` inside"}\nno closing fence here'
        assert _strip_fences(text) == text.strip()

    def test_multiline_json_in_fences(self):
        inner = '{\n  "a": 1,\n  "b": 2\n}'
        text = f"```json\n{inner}\n```"
        assert _strip_fences(text) == inner

    def test_backticks_inside_json_string_preserved(self):
        inner = '{"text": "use ```code``` here"}'
        # Closing fence is on its own line — inner content preserved
        text = f"```json\n{inner}\n```"
        assert _strip_fences(text) == inner


# --- _parse_json ---


class TestParseJson:
    def test_clean_json(self):
        assert _parse_json('{"key": "value"}') == {"key": "value"}

    def test_json_in_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert _parse_json(text) == {"key": "value"}

    def test_json_with_preamble(self):
        text = 'Here is the result:\n{"key": "value"}'
        assert _parse_json(text) == {"key": "value"}

    def test_json_with_preamble_and_postscript(self):
        text = 'Sure! Here you go:\n{"key": "value"}\nHope that helps!'
        assert _parse_json(text) == {"key": "value"}

    def test_nested_json(self):
        data = {"outer": {"inner": [1, 2, 3]}}
        assert _parse_json(json.dumps(data)) == data

    def test_no_json_raises(self):
        with pytest.raises(JsonParseError) as exc_info:
            _parse_json("This is just plain text with no JSON")
        assert "No JSON object found" in str(exc_info.value)
        assert exc_info.value.raw_output == "This is just plain text with no JSON"

    def test_invalid_json_raises(self):
        with pytest.raises(JsonParseError):
            _parse_json("{broken json: }")

    def test_array_response_raises(self):
        with pytest.raises(JsonParseError) as exc_info:
            _parse_json("[1, 2, 3]")
        assert "Expected dict" in str(exc_info.value)

    def test_stray_braces_before_json(self):
        text = 'Note: use {curly braces} in templates.\n{"ok": true}'
        assert _parse_json(text) == {"ok": True}

    def test_stray_braces_after_json(self):
        text = '{"ok": true}\nExtra note: object key {x}'
        assert _parse_json(text) == {"ok": True}

    def test_unicode_json(self):
        data = {"name": "Никита", "city": "Берлин", "emoji": "🎉"}
        assert _parse_json(json.dumps(data, ensure_ascii=False)) == data

    def test_json_parse_error_exposes_cause(self):
        with pytest.raises(JsonParseError) as exc_info:
            _parse_json("no json here")
        assert exc_info.value.cause is not None


# --- run_claude ---


class TestRunClaude:
    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_basic_call(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.return_value = MagicMock(stdout="  hello world  \n")
        result = run_claude("test prompt")
        assert result == "hello world"

        args = mock_run.call_args
        cmd = args[0][0]
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--setting-sources" in cmd
        assert "local" in cmd
        assert "test prompt" in cmd
        assert args[1]["encoding"] == "utf-8"
        assert args[1]["check"] is True
        assert args[1]["capture_output"] is True

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_model_arg(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.return_value = MagicMock(stdout="ok")
        run_claude("prompt", model="claude-sonnet-4-6")

        cmd = mock_run.call_args[0][0]
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "claude-sonnet-4-6"

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_timeout_passed(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.return_value = MagicMock(stdout="ok")
        run_claude("prompt", timeout=120)
        assert mock_run.call_args[1]["timeout"] == 120

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_env_strips_claudecode(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.return_value = MagicMock(stdout="ok")
        with patch.dict("os.environ", {"CLAUDECODE": "1", "HOME": "/home/test"}):
            run_claude("prompt")
        env = mock_run.call_args[1]["env"]
        assert "CLAUDECODE" not in env
        assert "HOME" in env

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_timeout_raises(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=60)
        with pytest.raises(ClaudeTimeoutError, match="timed out"):
            run_claude("prompt")

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_nonzero_exit_raises(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["claude"], stderr="something broke"
        )
        with pytest.raises(ClaudeError, match="something broke"):
            run_claude("prompt")

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_nonzero_exit_empty_stderr(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["claude"], stderr=""
        )
        with pytest.raises(ClaudeError, match="exit 1"):
            run_claude("prompt")

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_missing_cli_raises(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(ClaudeError, match="not found on PATH"):
            run_claude("prompt")

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_permission_error_raises(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.side_effect = PermissionError("Permission denied")
        with pytest.raises(ClaudeError, match="Failed to run"):
            run_claude("prompt")


# --- run_claude_json ---


class TestRunClaudeJson:
    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_returns_dict(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.return_value = MagicMock(stdout='{"shop": "EDEKA", "total": 12.50}')
        result = run_claude_json("extract receipt")
        assert result == {"shop": "EDEKA", "total": 12.50}

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_handles_fences(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.return_value = MagicMock(stdout='```json\n{"shop": "EDEKA", "total": 12.50}\n```')
        result = run_claude_json("extract receipt")
        assert result == {"shop": "EDEKA", "total": 12.50}

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_handles_preamble(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.return_value = MagicMock(
            stdout='Here is the extracted data:\n{"shop": "EDEKA", "total": 12.50}'
        )
        result = run_claude_json("extract receipt")
        assert result == {"shop": "EDEKA", "total": 12.50}

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_invalid_json_raises(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.return_value = MagicMock(stdout="I could not parse the receipt.")
        with pytest.raises(JsonParseError):
            run_claude_json("extract receipt")

    @patch("claude_runner.runner._ensure_nohooks_dir")
    @patch("claude_runner.runner.subprocess.run")
    def test_timeout_propagates(self, mock_run: MagicMock, mock_ensure: MagicMock):
        mock_ensure.return_value = "/tmp/nohooks"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=60)
        with pytest.raises(ClaudeTimeoutError):
            run_claude_json("extract receipt")
