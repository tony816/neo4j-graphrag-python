#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import json
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import ClaudeCodeLLM, CodexCLILLM
from neo4j_graphrag.llm.cli_llm import CLINotFoundError
from neo4j_graphrag.types import LLMMessage
from pydantic import BaseModel

CLAUDE_EXE = "/usr/local/bin/claude"
CODEX_EXE = "/usr/local/bin/codex"


class Answer(BaseModel):
    answer: str


def claude_payload(result: str = "hello", is_error: bool = False, **extra: Any) -> str:
    payload = {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "result": result,
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 5,
            "cache_creation_input_tokens": 0,
            "output_tokens": 3,
        },
    }
    payload.update(extra)
    return json.dumps(payload)


def codex_payload(text: str = "hello") -> str:
    return "\n".join(
        [
            '{"type":"thread.started","thread_id":"abc"}',
            "2026-01-01T00:00:00Z ERROR some unrelated log line",
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "item_0", "type": "agent_message", "text": text},
                }
            ),
            '{"type":"turn.completed","usage":{"input_tokens":21,"output_tokens":6}}',
        ]
    )


def run_result(stdout: str, stderr: str = "", returncode: int = 0) -> MagicMock:
    completed = MagicMock()
    completed.stdout = stdout
    completed.stderr = stderr
    completed.returncode = returncode
    return completed


def async_process(stdout: str, stderr: str = "", returncode: int = 0) -> MagicMock:
    process = MagicMock()
    process.communicate = AsyncMock(
        return_value=(stdout.encode("utf-8"), stderr.encode("utf-8"))
    )
    process.returncode = returncode
    return process


def call_args(mock_run: MagicMock) -> Tuple[List[str], Optional[str]]:
    """Returns the command and the prompt sent on stdin."""
    args, kwargs = mock_run.call_args
    return list(args[0]), kwargs.get("input")


# ---------------------------------------------------------------------------
# executable discovery
# ---------------------------------------------------------------------------


def test_find_executable_from_env_var(tmp_path: Path) -> None:
    fake = tmp_path / "claude"
    fake.write_text("", encoding="utf-8")
    with patch.dict("os.environ", {"CLAUDE_CLI_PATH": str(fake)}, clear=False):
        assert ClaudeCodeLLM.find_executable() == str(fake)


def test_find_executable_on_path() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with patch("shutil.which", return_value=CODEX_EXE):
            assert CodexCLILLM.find_executable() == CODEX_EXE


def test_find_executable_prefers_latest_version(tmp_path: Path) -> None:
    for version in ("1.9.0", "2.1.275", "2.1.9"):
        directory = tmp_path / version
        directory.mkdir()
        (directory / "claude.exe").write_text("", encoding="utf-8")
    with patch.dict("os.environ", {}, clear=True):
        with patch("shutil.which", return_value=None):
            with patch.object(
                ClaudeCodeLLM,
                "_candidate_globs",
                classmethod(lambda cls: (str(tmp_path / "*" / "claude.exe"),)),
            ):
                found = ClaudeCodeLLM.find_executable()
    assert found == str(tmp_path / "2.1.275" / "claude.exe")


def test_missing_executable_raises() -> None:
    with patch.object(ClaudeCodeLLM, "find_executable", return_value=None):
        with pytest.raises(CLINotFoundError, match="Could not find"):
            ClaudeCodeLLM()


# ---------------------------------------------------------------------------
# Claude Code CLI
# ---------------------------------------------------------------------------


@patch("subprocess.run")
def test_claude_invoke_builds_command_and_parses_result(mock_run: MagicMock) -> None:
    mock_run.return_value = run_result(claude_payload("Lady Jessica"))
    llm = ClaudeCodeLLM(model_name="sonnet", executable=CLAUDE_EXE)

    response = llm.invoke("Who is the mother of Paul Atreides?")

    command, prompt = call_args(mock_run)
    assert command[0] == CLAUDE_EXE
    assert command[1:4] == ["--print", "--output-format", "json"]
    assert "--model" in command and command[command.index("--model") + 1] == "sonnet"
    assert "--max-turns" in command
    assert "--restricted" in command
    assert "--strict-mcp-config" in command
    assert "--no-session-persistence" in command
    assert prompt == "Who is the mother of Paul Atreides?"
    assert response.content == "Lady Jessica"
    assert response.usage is not None
    assert response.usage.request_tokens == 15
    assert response.usage.response_tokens == 3
    assert response.usage.total_tokens == 18


@patch("subprocess.run")
def test_claude_invoke_with_message_history_and_system_instruction(
    mock_run: MagicMock,
) -> None:
    mock_run.return_value = run_result(claude_payload())
    llm = ClaudeCodeLLM(executable=CLAUDE_EXE)
    history: List[LLMMessage] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    llm.invoke("and now?", history, system_instruction="be brief")

    command, prompt = call_args(mock_run)
    assert command[command.index("--system-prompt") + 1] == "be brief"
    assert prompt is not None
    assert "<assistant>\nhello\n</assistant>" in prompt
    assert prompt.endswith("and now?")


@patch("subprocess.run")
def test_claude_append_system_prompt_mode(mock_run: MagicMock) -> None:
    mock_run.return_value = run_result(claude_payload())
    llm = ClaudeCodeLLM(executable=CLAUDE_EXE, system_prompt_mode="append")

    llm.invoke(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    )

    command, _ = call_args(mock_run)
    assert "--append-system-prompt" in command
    assert "--system-prompt" not in command


def test_claude_rejects_unknown_system_prompt_mode() -> None:
    with pytest.raises(ValueError, match="system_prompt_mode"):
        ClaudeCodeLLM(executable=CLAUDE_EXE, system_prompt_mode="nope")  # type: ignore[arg-type]


@patch("subprocess.run")
def test_claude_structured_output_passes_schema(mock_run: MagicMock) -> None:
    mock_run.return_value = run_result(
        claude_payload('Here it is:\n```json\n{"answer": "42"}\n```')
    )
    llm = ClaudeCodeLLM(executable=CLAUDE_EXE)

    response = llm.invoke(
        [{"role": "user", "content": "the answer?"}], response_format=Answer
    )

    command, _ = call_args(mock_run)
    schema = json.loads(command[command.index("--json-schema") + 1])
    assert schema["properties"]["answer"]["type"] == "string"
    assert schema["additionalProperties"] is False
    assert json.loads(response.content) == {"answer": "42"}


@patch("subprocess.run")
def test_claude_not_logged_in_raises_with_hint(mock_run: MagicMock) -> None:
    mock_run.return_value = run_result(
        claude_payload("Not logged in · Please run /login", is_error=True)
    )
    llm = ClaudeCodeLLM(executable=CLAUDE_EXE)

    with pytest.raises(LLMGenerationError, match="claude /login"):
        llm.invoke("hi")


@patch("subprocess.run")
def test_claude_non_json_output_raises(mock_run: MagicMock) -> None:
    mock_run.return_value = run_result("boom", stderr="unknown option", returncode=1)
    llm = ClaudeCodeLLM(executable=CLAUDE_EXE)

    with pytest.raises(LLMGenerationError, match="unknown option"):
        llm.invoke("hi")


@patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=1))
def test_claude_timeout_raises(mock_run: MagicMock) -> None:
    llm = ClaudeCodeLLM(executable=CLAUDE_EXE, timeout=1)

    with pytest.raises(LLMGenerationError, match="timed out"):
        llm.invoke("hi")


@pytest.mark.asyncio
async def test_claude_ainvoke() -> None:
    process = async_process(claude_payload("async answer"))
    with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
        llm = ClaudeCodeLLM(executable=CLAUDE_EXE)
        response = await llm.ainvoke("hi")

    assert response.content == "async answer"
    assert mock_exec.call_args[0][0] == CLAUDE_EXE
    process.communicate.assert_awaited_once_with(b"hi")


@patch("subprocess.run")
def test_claude_use_oauth_removes_api_key_from_child_env(mock_run: MagicMock) -> None:
    mock_run.return_value = run_result(claude_payload())
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}, clear=False):
        ClaudeCodeLLM(executable=CLAUDE_EXE).invoke("hi")
        env = mock_run.call_args.kwargs["env"]
        assert "ANTHROPIC_API_KEY" not in env

        ClaudeCodeLLM(executable=CLAUDE_EXE, use_oauth=False).invoke("hi")
        env = mock_run.call_args.kwargs["env"]
        assert env["ANTHROPIC_API_KEY"] == "sk-test"


def test_claude_auth_status_reads_oauth_file(tmp_path: Path) -> None:
    credentials = tmp_path / ".credentials.json"
    credentials.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "secret",
                    "expiresAt": 1893456000000,
                    "subscriptionType": "max",
                }
            }
        ),
        encoding="utf-8",
    )
    with patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": str(tmp_path)}, clear=False):
        status = ClaudeCodeLLM.auth_status()

    assert status.authenticated is True
    assert status.method == "oauth"
    assert status.account == "max"
    assert status.expires_at is not None
    assert "secret" not in status.model_dump_json()


def test_claude_auth_status_without_credentials(tmp_path: Path) -> None:
    with patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": str(tmp_path)}, clear=True):
        status = ClaudeCodeLLM.auth_status()

    assert status.authenticated is False
    assert "claude /login" in status.detail


def test_claude_auth_status_falls_back_to_api_key(tmp_path: Path) -> None:
    environment = {"CLAUDE_CONFIG_DIR": str(tmp_path), "ANTHROPIC_API_KEY": "sk-test"}
    with patch.dict("os.environ", environment, clear=True):
        status = ClaudeCodeLLM.auth_status()

    assert status.authenticated is True
    assert status.method == "api_key"
    assert "sk-test" not in status.model_dump_json()


# ---------------------------------------------------------------------------
# Codex CLI
# ---------------------------------------------------------------------------


@patch("subprocess.run")
def test_codex_invoke_builds_command_and_parses_events(mock_run: MagicMock) -> None:
    mock_run.return_value = run_result(codex_payload("Lady Jessica"))
    llm = CodexCLILLM(model_name="gpt-5-codex", executable=CODEX_EXE)

    response = llm.invoke("Who is the mother of Paul Atreides?")

    command, prompt = call_args(mock_run)
    assert command[:2] == [CODEX_EXE, "exec"]
    assert "--json" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-5-codex"
    assert "-c" in command and "mcp_servers={}" in command
    assert command[-1] == "-", "the prompt must be read from stdin"
    assert prompt == "Who is the mother of Paul Atreides?"
    assert response.content == "Lady Jessica"
    assert response.usage is not None
    assert response.usage.request_tokens == 21
    assert response.usage.total_tokens == 27


@patch("subprocess.run")
def test_codex_system_instruction_is_prepended(mock_run: MagicMock) -> None:
    mock_run.return_value = run_result(codex_payload())
    llm = CodexCLILLM(executable=CODEX_EXE)

    llm.invoke("hi", system_instruction="be brief")

    command, prompt = call_args(mock_run)
    assert "--system-prompt" not in command
    assert prompt is not None
    assert prompt.startswith("<system_instructions>\nbe brief\n</system_instructions>")
    assert prompt.endswith("hi")


@patch("subprocess.run")
def test_codex_structured_output_writes_schema_file(mock_run: MagicMock) -> None:
    written: dict[str, Any] = {}

    def capture(*args: Any, **kwargs: Any) -> MagicMock:
        command = list(args[0])
        schema_path = Path(command[command.index("--output-schema") + 1])
        written.update(json.loads(schema_path.read_text(encoding="utf-8")))
        return run_result(codex_payload('{"answer": "42"}'))

    mock_run.side_effect = capture
    llm = CodexCLILLM(executable=CODEX_EXE)

    response = llm.invoke(
        [{"role": "user", "content": "the answer?"}], response_format=Answer
    )

    assert written["properties"]["answer"]["type"] == "string"
    assert json.loads(response.content) == {"answer": "42"}


@patch("subprocess.run")
def test_codex_error_event_raises(mock_run: MagicMock) -> None:
    mock_run.return_value = run_result(
        '{"type":"error","message":"Please log in with `codex login`"}',
        returncode=1,
    )
    llm = CodexCLILLM(executable=CODEX_EXE)

    with pytest.raises(LLMGenerationError, match="codex login"):
        llm.invoke("hi")


@patch("subprocess.run")
def test_codex_without_agent_message_raises(mock_run: MagicMock) -> None:
    mock_run.return_value = run_result('{"type":"turn.started"}')
    llm = CodexCLILLM(executable=CODEX_EXE)

    with pytest.raises(LLMGenerationError, match="no assistant message"):
        llm.invoke("hi")


@patch("subprocess.run")
def test_codex_parses_legacy_event_shape(mock_run: MagicMock) -> None:
    mock_run.return_value = run_result(
        "\n".join(
            [
                '{"msg":{"type":"agent_message","message":"legacy"}}',
                '{"msg":{"type":"token_count","info":{"total_token_usage":'
                '{"input_tokens":4,"output_tokens":2}}}}',
            ]
        )
    )
    llm = CodexCLILLM(executable=CODEX_EXE)

    response = llm.invoke("hi")

    assert response.content == "legacy"
    assert response.usage is not None
    assert response.usage.total_tokens == 6


@pytest.mark.asyncio
async def test_codex_ainvoke() -> None:
    process = async_process(codex_payload("async answer"))
    with patch("asyncio.create_subprocess_exec", return_value=process):
        llm = CodexCLILLM(executable=CODEX_EXE)
        response = await llm.ainvoke([{"role": "user", "content": "hi"}])

    assert response.content == "async answer"


def test_codex_auth_status_reads_oauth_file(tmp_path: Path) -> None:
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": {"access_token": "secret", "account_id": "acct_1"},
                "last_refresh": "2026-09-18T01:31:49.322068900Z",
            }
        ),
        encoding="utf-8",
    )
    with patch.dict("os.environ", {"CODEX_HOME": str(tmp_path)}, clear=False):
        status = CodexCLILLM.auth_status()

    assert status.authenticated is True
    assert status.method == "oauth"
    assert status.account == "acct_1"
    assert status.expires_at is not None
    assert "secret" not in status.model_dump_json()


def test_codex_auth_status_when_not_logged_in(tmp_path: Path) -> None:
    environment = {"CODEX_HOME": str(tmp_path)}
    with patch.dict("os.environ", environment, clear=True):
        status = CodexCLILLM.auth_status()

    assert status.authenticated is False
    assert "codex login" in status.detail
