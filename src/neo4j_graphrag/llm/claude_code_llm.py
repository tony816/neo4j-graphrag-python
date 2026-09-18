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
"""LLM provider backed by the Claude Code CLI installed on the machine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

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


class ClaudeCodeLLM(BaseCLILLM):
    """Interface to Claude through the locally installed Claude Code CLI.

    The CLI is run in non-interactive mode (``claude --print``) and authenticates
    with the OAuth session created by ``claude /login``; no API key is required
    and no token is read by this library. Subscription plans (Pro/Max) therefore
    work out of the box, subject to their usage limits and terms.

    Args:
        model_name (str): Model alias or full name, e.g. ``"sonnet"``,
            ``"opus"`` or ``"claude-sonnet-5"``. Empty (default) leaves the
            choice to the CLI configuration.
        model_params (Optional[dict[str, Any]]): Unused, kept for interface
            compatibility.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate
            limiting. Defaults to retry with exponential backoff.
        executable (Optional[str]): Path to the ``claude`` binary. Auto-detected
            when omitted (PATH, ``CLAUDE_CLI_PATH``, then the usual install
            directories).
        cwd (Optional[str]): Working directory for the CLI process. Defaults to
            a temporary directory, so project files are not picked up.
        timeout (Optional[float]): Seconds allowed per call. Defaults to 600.
        env (Optional[Mapping[str, str]]): Extra environment variables.
        use_oauth (bool): Strip ``ANTHROPIC_API_KEY``/``ANTHROPIC_AUTH_TOKEN``
            from the child environment so the stored OAuth session is used.
            Defaults to True.
        system_prompt_mode (str): ``"replace"`` (default) passes the system
            instruction with ``--system-prompt``, so the model behaves like a
            plain assistant instead of a coding agent; ``"append"`` keeps the
            Claude Code system prompt and appends to it.
        max_turns (Optional[int]): Value for ``--max-turns``. Defaults to 1 so a
            call cannot turn into a long agentic session. None omits the flag.
        restricted (bool): Run with ``--restricted --permission-prompts none``,
            which removes the command-running tools and denies anything that
            would prompt. Defaults to True.
        allow_mcp (bool): Keep the MCP servers configured in the CLI. Defaults
            to False (``--strict-mcp-config``).
        persist_session (bool): Save each call as a resumable CLI session.
            Defaults to False (``--no-session-persistence``).
        extra_args (Optional[Sequence[str]]): Extra CLI arguments, appended
            last, e.g. ``["--add-dir", "/data"]``.

    Raises:
        CLINotFoundError: If the CLI cannot be located.
        LLMGenerationError: If the CLI fails, times out or is not logged in.

    Example:

    .. code-block:: python

        from neo4j_graphrag.llm import ClaudeCodeLLM

        llm = ClaudeCodeLLM(model_name="sonnet")
        print(llm.invoke("Who is the mother of Paul Atreides?").content)
    """

    cli_name: ClassVar[str] = "claude"
    display_name: ClassVar[str] = "Claude Code CLI"
    executable_names: ClassVar[tuple[str, ...]] = ("claude",)
    executable_env_vars: ClassVar[tuple[str, ...]] = (
        "NEO4J_GRAPHRAG_CLAUDE_CLI",
        "CLAUDE_CLI_PATH",
    )
    login_command: ClassVar[str] = "claude /login"
    api_key_env_vars: ClassVar[tuple[str, ...]] = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
    )

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
        system_prompt_mode: Literal["replace", "append"] = "replace",
        max_turns: Optional[int] = 1,
        restricted: bool = True,
        allow_mcp: bool = False,
        persist_session: bool = False,
        extra_args: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        if system_prompt_mode not in ("replace", "append"):
            raise ValueError(
                f"system_prompt_mode must be 'replace' or 'append', got {system_prompt_mode!r}"
            )
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
        self.system_prompt_mode = system_prompt_mode
        self.max_turns = max_turns
        self.restricted = restricted
        self.allow_mcp = allow_mcp
        self.persist_session = persist_session

    @classmethod
    def _candidate_globs(cls) -> Sequence[str]:
        home = _home()
        appdata = os.environ.get("APPDATA") or str(home / "AppData" / "Roaming")
        localappdata = os.environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        return (
            # Native installer / npm global install.
            str(home / ".claude" / "local" / "claude"),
            str(home / ".local" / "bin" / "claude"),
            str(home / ".bun" / "bin" / "claude"),
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
            # Windows: desktop app and npm global installs.
            os.path.join(appdata, "npm", "claude.cmd"),
            os.path.join(appdata, "Claude", "claude-code", "*", "claude.exe"),
            os.path.join(
                localappdata,
                "Packages",
                "Claude_*",
                "LocalCache",
                "Roaming",
                "Claude",
                "claude-code",
                "*",
                "claude.exe",
            ),
        )

    @classmethod
    def auth_status(cls) -> CLIAuthStatus:
        """Reports what the Claude Code credential store holds.

        On macOS (Keychain) and Windows (Credential Manager) the token may not
        be readable from a file, in which case ``authenticated`` is False while
        the CLI still works: it is a hint, not a gate.
        """
        if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return CLIAuthStatus(
                cli=cls.cli_name,
                authenticated=True,
                method="oauth",
                detail="Using the OAuth token from CLAUDE_CODE_OAUTH_TOKEN.",
            )
        config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR") or _home() / ".claude")
        path = config_dir / ".credentials.json"
        data = _read_json_file(path) or {}
        oauth = data.get("claudeAiOauth")
        if isinstance(oauth, dict) and oauth.get("accessToken"):
            return CLIAuthStatus(
                cli=cls.cli_name,
                authenticated=True,
                method="oauth",
                account=oauth.get("subscriptionType"),
                expires_at=_parse_timestamp(oauth.get("expiresAt")),
                credentials_path=str(path),
                detail="Signed in with a Claude account (OAuth).",
            )
        if os.environ.get("ANTHROPIC_API_KEY"):
            return CLIAuthStatus(
                cli=cls.cli_name,
                authenticated=True,
                method="api_key",
                detail=(
                    "No OAuth session found, but ANTHROPIC_API_KEY is set. Pass "
                    "use_oauth=False to let the CLI use it."
                ),
            )
        return CLIAuthStatus(
            cli=cls.cli_name,
            authenticated=False,
            credentials_path=str(path),
            detail=(
                "No OAuth session found in the credential file. On macOS and Windows the "
                "token is kept in the OS keychain, so the CLI may still be logged in; "
                "run `claude /login` if calls fail."
            ),
        )

    def _build_args(
        self,
        system_instruction: Optional[str],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
        workdir: Path,
    ) -> List[str]:
        args = ["--print", "--output-format", "json"]
        if self.model_name:
            args += ["--model", self.model_name]
        if self.max_turns is not None:
            args += ["--max-turns", str(self.max_turns)]
        if self.restricted:
            args += ["--restricted", "--permission-prompts", "none"]
        if not self.allow_mcp:
            args.append("--strict-mcp-config")
        if not self.persist_session:
            args.append("--no-session-persistence")
        if system_instruction:
            flag = (
                "--system-prompt"
                if self.system_prompt_mode == "replace"
                else "--append-system-prompt"
            )
            args += [flag, system_instruction]
        schema = self._json_schema(response_format)
        if schema is not None:
            args += ["--json-schema", json.dumps(schema)]
        args += self.extra_args
        return args

    def _parse_response(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
    ) -> LLMResponse:
        try:
            payload = json.loads(stdout.strip() or "{}")
        except json.JSONDecodeError:
            if returncode != 0:
                raise self._fail(stderr or stdout or f"exit code {returncode}")
            raise LLMGenerationError(
                f"Could not parse the {self.display_name} output as JSON: {stdout[:500]}"
            )
        if not isinstance(payload, dict):
            raise LLMGenerationError(
                f"Unexpected {self.display_name} output: {stdout[:500]}"
            )
        result = payload.get("result") or ""
        if returncode != 0 or payload.get("is_error"):
            raise self._fail(result or stderr or f"exit code {returncode}")
        if not result:
            raise LLMGenerationError(f"{self.display_name} returned an empty response.")
        return LLMResponse(
            content=self._postprocess_content(result, response_format),
            usage=self._usage(payload.get("usage")),
        )

    @staticmethod
    def _usage(usage: Any) -> Optional[LLMUsage]:
        if not isinstance(usage, dict):
            return None
        request_tokens = sum(
            value
            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
            if isinstance(value := usage.get(key), int)
        )
        response_tokens = usage.get("output_tokens")
        if not isinstance(response_tokens, int):
            response_tokens = None
        if not request_tokens and response_tokens is None:
            return None
        return LLMUsage(
            request_tokens=request_tokens or None,
            response_tokens=response_tokens,
            total_tokens=(request_tokens + response_tokens)
            if (request_tokens and response_tokens is not None)
            else None,
        )
