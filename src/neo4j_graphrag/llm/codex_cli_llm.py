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
"""LLM provider backed by the Codex CLI installed on the machine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar, List, Mapping, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.utils.rate_limit import RateLimitHandler

from .cli_llm import (
    DEFAULT_CLI_TIMEOUT,
    BaseCLILLM,
    CLIAuthStatus,
    _home,
    _parse_timestamp,
    _read_json_file,
)
from .types import LLMResponse, LLMUsage

# pylint: disable=redefined-builtin, arguments-differ


class CodexCLILLM(BaseCLILLM):
    """Interface to OpenAI models through the locally installed Codex CLI.

    The CLI is run as ``codex exec`` and authenticates with the ChatGPT OAuth
    session created by ``codex login``; no API key is required and no token is
    read by this library. ChatGPT plans (Plus/Pro/Business) therefore work out
    of the box, subject to their usage limits and terms.

    Codex has no dedicated system-prompt flag, so a system instruction is
    prepended to the prompt.

    Args:
        model_name (str): Model to use, e.g. ``"gpt-5-codex"``. Empty (default)
            leaves the choice to the CLI configuration.
        model_params (Optional[dict[str, Any]]): Unused, kept for interface
            compatibility.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate
            limiting. Defaults to retry with exponential backoff.
        executable (Optional[str]): Path to the ``codex`` binary. Auto-detected
            when omitted (PATH, ``CODEX_CLI_PATH``, then the usual install
            directories).
        cwd (Optional[str]): Working directory for the CLI process. Defaults to
            a temporary directory, so project files are not picked up.
        timeout (Optional[float]): Seconds allowed per call. Defaults to 600.
        env (Optional[Mapping[str, str]]): Extra environment variables.
        use_oauth (bool): Strip ``OPENAI_API_KEY`` from the child environment
            and ask the CLI for the ChatGPT sign-in. Defaults to True.
        sandbox (str): Sandbox policy for commands the agent may try to run:
            ``"read-only"`` (default), ``"workspace-write"`` or
            ``"danger-full-access"``.
        allow_mcp (bool): Keep the MCP servers configured in the CLI. Defaults
            to False, which starts the call faster.
        config_overrides (Optional[Mapping[str, str]]): Extra ``-c key=value``
            configuration overrides.
        extra_args (Optional[Sequence[str]]): Extra CLI arguments, appended
            after the options and before the prompt marker.

    Raises:
        CLINotFoundError: If the CLI cannot be located.
        LLMGenerationError: If the CLI fails, times out or is not logged in.

    Example:

    .. code-block:: python

        from neo4j_graphrag.llm import CodexCLILLM

        llm = CodexCLILLM(model_name="gpt-5-codex")
        print(llm.invoke("Who is the mother of Paul Atreides?").content)
    """

    cli_name: ClassVar[str] = "codex"
    display_name: ClassVar[str] = "Codex CLI"
    executable_names: ClassVar[tuple[str, ...]] = ("codex",)
    executable_env_vars: ClassVar[tuple[str, ...]] = (
        "NEO4J_GRAPHRAG_CODEX_CLI",
        "CODEX_CLI_PATH",
    )
    login_command: ClassVar[str] = "codex login"
    api_key_env_vars: ClassVar[tuple[str, ...]] = ("OPENAI_API_KEY",)

    def __init__(
        self,
        model_name: str = "",
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        executable: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_CLI_TIMEOUT,
        env: Optional[Mapping[str, str]] = None,
        use_oauth: bool = True,
        sandbox: str = "read-only",
        allow_mcp: bool = False,
        config_overrides: Optional[Mapping[str, str]] = None,
        extra_args: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_params=model_params,
            rate_limit_handler=rate_limit_handler,
            executable=executable,
            cwd=cwd,
            timeout=timeout,
            env=env,
            use_oauth=use_oauth,
            extra_args=extra_args,
            **kwargs,
        )
        self.sandbox = sandbox
        self.allow_mcp = allow_mcp
        self.config_overrides = dict(config_overrides or {})

    @classmethod
    def _candidate_globs(cls) -> Sequence[str]:
        home = _home()
        appdata = os.environ.get("APPDATA") or str(home / "AppData" / "Roaming")
        localappdata = os.environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        codex_home = os.environ.get("CODEX_HOME") or str(home / ".codex")
        return (
            os.path.join(codex_home, "bin", "codex"),
            str(home / ".local" / "bin" / "codex"),
            str(home / ".bun" / "bin" / "codex"),
            "/usr/local/bin/codex",
            "/opt/homebrew/bin/codex",
            # Windows: desktop app and npm global installs.
            os.path.join(appdata, "npm", "codex.cmd"),
            os.path.join(localappdata, "OpenAI", "Codex", "bin", "*", "codex.exe"),
        )

    @classmethod
    def auth_status(cls) -> CLIAuthStatus:
        """Reports what ``$CODEX_HOME/auth.json`` holds (never the tokens)."""
        codex_home = Path(os.environ.get("CODEX_HOME") or _home() / ".codex")
        path = codex_home / "auth.json"
        data = _read_json_file(path) or {}
        tokens = data.get("tokens")
        if isinstance(tokens, dict) and tokens.get("access_token"):
            return CLIAuthStatus(
                cli=cls.cli_name,
                authenticated=True,
                method="oauth",
                account=tokens.get("account_id"),
                expires_at=_parse_timestamp(data.get("last_refresh")),
                credentials_path=str(path),
                detail="Signed in with a ChatGPT account (OAuth).",
            )
        if data.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            return CLIAuthStatus(
                cli=cls.cli_name,
                authenticated=True,
                method="api_key",
                credentials_path=str(path) if data.get("OPENAI_API_KEY") else None,
                detail=(
                    "No ChatGPT OAuth session found, but an API key is available. "
                    "Pass use_oauth=False to let the CLI use the key from the "
                    "environment."
                ),
            )
        return CLIAuthStatus(
            cli=cls.cli_name,
            authenticated=False,
            credentials_path=str(path),
            detail="Not signed in. Run `codex login` once in a terminal.",
        )

    def _prompt_prefix(self, system_instruction: Optional[str]) -> Optional[str]:
        # `codex exec` has no system prompt flag.
        if not system_instruction:
            return None
        return f"<system_instructions>\n{system_instruction}\n</system_instructions>"

    def _build_args(
        self,
        system_instruction: Optional[str],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
        workdir: Path,
    ) -> List[str]:
        args = [
            "exec",
            "--json",
            "--color",
            "never",
            "--skip-git-repo-check",
            "--sandbox",
            self.sandbox,
        ]
        if self.model_name:
            args += ["--model", self.model_name]
        overrides = dict(self.config_overrides)
        if not self.allow_mcp:
            overrides.setdefault("mcp_servers", "{}")
        if self.use_oauth:
            overrides.setdefault("preferred_auth_method", '"chatgpt"')
        for key, value in overrides.items():
            args += ["-c", f"{key}={value}"]
        schema = self._json_schema(response_format)
        if schema is not None:
            schema_file = workdir / "output_schema.json"
            schema_file.write_text(json.dumps(schema), encoding="utf-8")
            args += ["--output-schema", str(schema_file)]
        args += self.extra_args
        # Read the prompt from stdin: prompts are too long for a command line.
        args.append("-")
        return args

    def _parse_response(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
    ) -> LLMResponse:
        message, usage, error = self._parse_events(stdout)
        if returncode != 0:
            raise self._fail(error or stderr or stdout or f"exit code {returncode}")
        if error and not message:
            raise self._fail(error)
        if not message:
            raise LLMGenerationError(
                f"{self.display_name} returned no assistant message. "
                f"stderr: {stderr.strip()[:500]}"
            )
        return LLMResponse(
            content=self._postprocess_content(message, response_format), usage=usage
        )

    @staticmethod
    def _parse_events(
        stdout: str,
    ) -> Tuple[Optional[str], Optional[LLMUsage], Optional[str]]:
        """Extracts the final message, usage and error from the JSONL stream.

        Lines that are not JSON (the CLI also logs to stdout on occasion) are
        ignored.
        """
        message: Optional[str] = None
        usage: Optional[LLMUsage] = None
        error: Optional[str] = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            # Older CLI versions nest the payload under "msg".
            inner = event.get("msg")
            if isinstance(inner, dict):
                event = {**inner, "type": inner.get("type")}
            event_type = event.get("type")
            if event_type == "item.completed":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        message = text
            elif event_type == "agent_message":
                text = event.get("message") or event.get("text")
                if isinstance(text, str) and text:
                    message = text
            elif event_type == "turn.completed":
                usage = CodexCLILLM._usage(event.get("usage")) or usage
            elif event_type == "token_count":
                info = event.get("info")
                if isinstance(info, dict):
                    usage = CodexCLILLM._usage(info.get("total_token_usage")) or usage
            elif event_type in ("error", "turn.failed", "stream_error"):
                detail = event.get("error") or event.get("message")
                if isinstance(detail, dict):
                    detail = detail.get("message")
                if isinstance(detail, str) and detail:
                    error = detail
        return message, usage, error

    @staticmethod
    def _usage(usage: Any) -> Optional[LLMUsage]:
        if not isinstance(usage, dict):
            return None
        request_tokens = usage.get("input_tokens")
        response_tokens = usage.get("output_tokens")
        if not isinstance(request_tokens, int):
            request_tokens = None
        if not isinstance(response_tokens, int):
            response_tokens = None
        if request_tokens is None and response_tokens is None:
            return None
        return LLMUsage(
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            total_tokens=(request_tokens + response_tokens)
            if (request_tokens is not None and response_tokens is not None)
            else None,
        )
