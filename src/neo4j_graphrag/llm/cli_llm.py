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
"""Shared plumbing for LLM providers backed by a locally installed agent CLI.

These providers do not call a vendor HTTP API directly: they spawn the CLI that
is already installed on the machine (Claude Code, Codex) in non-interactive mode
and let it authenticate with the OAuth session created by ``claude /login`` /
``codex login``. This library never reads, copies or forwards a token: the CLI
reads its own credential store and refreshes it when needed.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from pydantic import BaseModel, ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
)
from neo4j_graphrag.utils.rate_limit import (
    async_rate_limit_handler as async_rate_limit_handler_decorator,
)
from neo4j_graphrag.utils.rate_limit import (
    rate_limit_handler as rate_limit_handler_decorator,
)

from .anthropic_llm import BaseAnthropicLLM, _to_anthropic_schema
from .base import LLMBase
from .types import BaseMessage, LLMResponse, MessageList

# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from

logger = logging.getLogger(__name__)

DEFAULT_CLI_TIMEOUT = 600.0
"""Default number of seconds to wait for a single CLI invocation."""

_VERSION_PART = re.compile(r"\d+")
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class CLIAuthStatus(BaseModel):
    """Result of inspecting the credential store of a local agent CLI.

    Only metadata is exposed: access and refresh tokens are never read into this
    object.

    Attributes:
        cli (str): Name of the CLI that was inspected, e.g. ``"claude"``.
        authenticated (bool): Whether usable credentials were found.
        method (Optional[str]): ``"oauth"``, ``"api_key"`` or ``None`` when
            undetermined.
        account (Optional[str]): Account identifier or subscription type
            reported by the credential store, when available.
        expires_at (Optional[datetime]): Expiry of the OAuth access token, when
            the credential store records one. An expired access token is not a
            problem in itself: the CLI refreshes it on the next call.
        credentials_path (Optional[str]): File the information was read from.
        detail (str): Human readable summary, suitable for logging or for
            telling the user what to do next.
    """

    cli: str
    authenticated: bool
    method: Optional[str] = None
    account: Optional[str] = None
    expires_at: Optional[datetime] = None
    credentials_path: Optional[str] = None
    detail: str = ""


class CLINotFoundError(LLMGenerationError):
    """Raised when the agent CLI cannot be located on the machine."""


def _version_sort_key(path: str) -> Tuple[Tuple[int, ...], float]:
    """Sort key ranking installation paths newest-first.

    Most installers drop the binary in a version-named directory
    (``.../claude-code/2.1.275/claude.exe``), so the numbers found in the parent
    directory name are used first, with the file mtime as a tie-breaker.
    """
    parent = os.path.basename(os.path.dirname(path))
    numbers = tuple(int(part) for part in _VERSION_PART.findall(parent))
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return numbers, mtime


def _home() -> Path:
    """Home directory, without raising when it cannot be determined."""
    return Path(os.path.expanduser("~"))


def _read_json_file(path: Path) -> Optional[dict[str, Any]]:
    """Reads a JSON object from *path*, returning None when unreadable."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parses the timestamp formats used by the CLI credential stores."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        # Claude Code stores milliseconds since epoch.
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    if isinstance(value, str) and value:
        # Codex writes RFC 3339 with nanoseconds, which fromisoformat rejects
        # before Python 3.11.
        text = re.sub(r"(\.\d{6})\d+", r"\1", value.replace("Z", "+00:00"))
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


class BaseCLILLM(LLMBase, ABC):
    """Base class for LLMs served by a locally installed agent CLI.

    Subclasses describe how to find the executable, how to turn a prompt into a
    command line, and how to read the CLI output back.

    Args:
        model_name (str): Model passed to the CLI. An empty string means
            "whatever the CLI is configured to use".
        model_params (Optional[dict[str, Any]]): Unused by CLI providers, kept
            for interface compatibility.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate
            limiting. Defaults to retry with exponential backoff.
        executable (Optional[str]): Path to the CLI binary. When omitted, it is
            auto-discovered (see :meth:`find_executable`).
        cwd (Optional[str]): Working directory for the CLI process. Defaults to
            a temporary directory so the CLI does not pick up the project it is
            called from (``CLAUDE.md``, ``AGENTS.md``, git state...).
        timeout (Optional[float]): Seconds to wait for one invocation. ``None``
            disables the timeout. Defaults to 600.
        env (Optional[Mapping[str, str]]): Extra environment variables for the
            CLI process.
        use_oauth (bool): When True (default), API-key environment variables are
            removed from the child environment so the CLI uses the OAuth session
            from its own credential store rather than a key that happens to be
            exported in the parent process.
        extra_args (Optional[Sequence[str]]): Additional command line arguments
            appended to every invocation.
    """

    supports_structured_output: bool = True

    cli_name: ClassVar[str] = ""
    """Name of the CLI, used in error messages."""

    display_name: ClassVar[str] = ""
    """Human readable name of the CLI."""

    executable_names: ClassVar[Tuple[str, ...]] = ()
    """Names looked up on PATH."""

    executable_env_vars: ClassVar[Tuple[str, ...]] = ()
    """Environment variables that can point at the executable."""

    login_command: ClassVar[str] = ""
    """Command the user must run to (re-)authenticate."""

    api_key_env_vars: ClassVar[Tuple[str, ...]] = ()
    """Environment variables removed from the child env when ``use_oauth``."""

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
        extra_args: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        LLMBase.__init__(
            self,
            model_name=model_name,
            model_params=model_params or {},
            rate_limit_handler=rate_limit_handler,
            **kwargs,
        )
        resolved = executable or self.find_executable()
        if resolved is None:
            raise CLINotFoundError(
                f"Could not find the {self.display_name} executable. Install it, put it "
                f"on PATH, set one of {', '.join(self.executable_env_vars)}, or pass "
                f"executable='/path/to/{self.cli_name}'."
            )
        self.executable = resolved
        self.cwd = cwd or tempfile.gettempdir()
        self.timeout = timeout
        self.env = dict(env or {})
        self.use_oauth = use_oauth
        self.extra_args = list(extra_args or [])

    # ------------------------------------------------------------------
    # discovery and authentication
    # ------------------------------------------------------------------

    @classmethod
    def _candidate_globs(cls) -> Sequence[str]:
        """Glob patterns for well-known installation directories."""
        return ()

    @classmethod
    def find_executable(cls) -> Optional[str]:
        """Locates the CLI binary.

        Lookup order: the environment variables in ``executable_env_vars``, then
        PATH, then the platform specific installation directories returned by
        ``_candidate_globs`` (newest version first).

        Returns:
            Optional[str]: Path to the executable, or None when not found.
        """
        for var in cls.executable_env_vars:
            value = os.environ.get(var)
            if not value:
                continue
            if os.path.isfile(value):
                return value
            on_path = shutil.which(value)
            if on_path:
                return on_path
            logger.warning(
                "%s is set to '%s' but no such executable could be found", var, value
            )
        for name in cls.executable_names:
            on_path = shutil.which(name)
            if on_path:
                return on_path
        candidates = [
            path
            for pattern in cls._candidate_globs()
            for path in glob.glob(os.path.expanduser(pattern))
            if os.path.isfile(path)
        ]
        if not candidates:
            return None
        return max(candidates, key=_version_sort_key)

    @classmethod
    def auth_status(cls) -> CLIAuthStatus:
        """Inspects the credential store of the CLI without exposing secrets.

        This is informational only: the CLI itself is the source of truth and
        performs token refresh. ``authenticated=False`` does not necessarily
        mean the CLI will fail, since some platforms keep credentials in an OS
        keychain this method cannot read.

        Returns:
            CLIAuthStatus: What was found in the credential store.
        """
        return CLIAuthStatus(
            cli=cls.cli_name,
            authenticated=False,
            detail="Authentication status cannot be determined for this CLI.",
        )

    def version(self) -> str:
        """Returns the version string reported by the CLI."""
        stdout, stderr, code = self._run_cli(["--version"], prompt=None)
        if code != 0:
            raise LLMGenerationError(
                f"'{self.executable} --version' exited with code {code}: {stderr.strip()}"
            )
        return stdout.strip()

    # ------------------------------------------------------------------
    # process handling
    # ------------------------------------------------------------------

    def _child_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.use_oauth:
            for var in self.api_key_env_vars:
                env.pop(var, None)
        env.update(self.env)
        return env

    def _popen_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"cwd": self.cwd, "env": self._child_env()}
        if sys.platform == "win32":
            # Keep the library usable from GUI apps: no console window flashes.
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        return kwargs

    def _run_cli(
        self, args: Sequence[str], prompt: Optional[str]
    ) -> Tuple[str, str, int]:
        command = [self.executable, *args]
        logger.debug("Running %s", command)
        try:
            completed = subprocess.run(
                command,
                input=prompt if prompt is not None else "",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                **self._popen_kwargs(),
            )
        except subprocess.TimeoutExpired as e:
            raise LLMGenerationError(
                f"{self.display_name} timed out after {self.timeout} seconds."
            ) from e
        except OSError as e:
            raise LLMGenerationError(
                f"Could not run {self.display_name} ('{self.executable}'): {e}"
            ) from e
        return completed.stdout or "", completed.stderr or "", completed.returncode

    async def _arun_cli(
        self, args: Sequence[str], prompt: Optional[str]
    ) -> Tuple[str, str, int]:
        command = [self.executable, *args]
        logger.debug("Running %s", command)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **self._popen_kwargs(),
            )
        except OSError as e:
            raise LLMGenerationError(
                f"Could not run {self.display_name} ('{self.executable}'): {e}"
            ) from e
        payload = (prompt or "").encode("utf-8")
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(payload), timeout=self.timeout
            )
        except asyncio.TimeoutError as e:
            process.kill()
            await process.wait()
            raise LLMGenerationError(
                f"{self.display_name} timed out after {self.timeout} seconds."
            ) from e
        return (
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            process.returncode if process.returncode is not None else -1,
        )

    # ------------------------------------------------------------------
    # subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_args(
        self,
        system_instruction: Optional[str],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
        workdir: Path,
    ) -> List[str]:
        """Builds the CLI arguments for one invocation.

        Args:
            system_instruction: System prompt for this call, if any.
            response_format: Requested structured output, if any.
            workdir: Scratch directory usable for temporary files (schemas...).
        """

    @abstractmethod
    def _parse_response(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
    ) -> LLMResponse:
        """Turns the CLI output into an :class:`LLMResponse`."""

    def _prompt_prefix(self, system_instruction: Optional[str]) -> Optional[str]:
        """System instruction to prepend to the prompt.

        Used by CLIs that have no dedicated system prompt flag. Returning None
        (the default) means the system instruction is passed on the command line
        by :meth:`_build_args` instead.
        """
        return None

    # ------------------------------------------------------------------
    # invocation
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return self.__invoke_v1(input, message_history, system_instruction)
        elif isinstance(input, list):
            return self.__invoke_v2(input, response_format=response_format, **kwargs)
        else:
            raise ValueError(f"Invalid input type for invoke method - {type(input)}")

    async def ainvoke(
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return await self.__ainvoke_v1(input, message_history, system_instruction)
        elif isinstance(input, list):
            return await self.__ainvoke_v2(
                input, response_format=response_format, **kwargs
            )
        else:
            raise ValueError(f"Invalid input type for ainvoke method - {type(input)}")

    @rate_limit_handler_decorator
    def __invoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        prompt = self._build_prompt(self._validated_history(message_history), input)
        return self._generate(prompt, system_instruction, None)

    @rate_limit_handler_decorator
    def __invoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system_instruction, prompt = self._split_messages(input)
        return self._generate(prompt, system_instruction, response_format)

    @async_rate_limit_handler_decorator
    async def __ainvoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        prompt = self._build_prompt(self._validated_history(message_history), input)
        return await self._agenerate(prompt, system_instruction, None)

    @async_rate_limit_handler_decorator
    async def __ainvoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system_instruction, prompt = self._split_messages(input)
        return await self._agenerate(prompt, system_instruction, response_format)

    def _generate(
        self,
        prompt: str,
        system_instruction: Optional[str],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
    ) -> LLMResponse:
        with tempfile.TemporaryDirectory(prefix="neo4j_graphrag_cli_") as tmp:
            args, prompt = self._prepare(
                prompt, system_instruction, response_format, Path(tmp)
            )
            stdout, stderr, code = self._run_cli(args, prompt)
            return self._parse_response(stdout, stderr, code, response_format)

    async def _agenerate(
        self,
        prompt: str,
        system_instruction: Optional[str],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
    ) -> LLMResponse:
        with tempfile.TemporaryDirectory(prefix="neo4j_graphrag_cli_") as tmp:
            args, prompt = self._prepare(
                prompt, system_instruction, response_format, Path(tmp)
            )
            stdout, stderr, code = await self._arun_cli(args, prompt)
            return self._parse_response(stdout, stderr, code, response_format)

    def _prepare(
        self,
        prompt: str,
        system_instruction: Optional[str],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
        workdir: Path,
    ) -> Tuple[List[str], str]:
        prefix = self._prompt_prefix(system_instruction)
        if prefix:
            prompt = f"{prefix}\n\n{prompt}"
            system_instruction = None
        args = self._build_args(system_instruction, response_format, workdir)
        return args, prompt

    # ------------------------------------------------------------------
    # prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _validated_history(
        message_history: Optional[Union[List[LLMMessage], MessageHistory]],
    ) -> List[LLMMessage]:
        if message_history is None:
            return []
        if isinstance(message_history, MessageHistory):
            message_history = message_history.messages
        try:
            MessageList(messages=cast(List[BaseMessage], message_history))
        except ValidationError as e:
            raise LLMGenerationError(e.errors()) from e
        return list(message_history)

    @staticmethod
    def _build_prompt(history: Sequence[LLMMessage], input: str) -> str:
        """Renders history and the current message as a single prompt.

        Agent CLIs are invoked once per call with a single prompt, so a
        multi-turn history is replayed as a transcript.
        """
        if not history:
            return input
        transcript = "\n\n".join(
            f"<{message['role']}>\n{message['content']}\n</{message['role']}>"
            for message in history
        )
        return (
            "Here is the conversation so far:\n\n"
            f"{transcript}\n\n"
            "Reply to this last message:\n\n"
            f"{input}"
        )

    @classmethod
    def _split_messages(cls, input: List[LLMMessage]) -> Tuple[Optional[str], str]:
        """Splits V2 messages into a system instruction and a single prompt."""
        system_parts: List[str] = []
        conversation: List[LLMMessage] = []
        for message in input:
            role = message["role"]
            if role == "system":
                system_parts.append(message["content"])
            elif role in ("user", "assistant"):
                conversation.append(message)
            else:
                raise ValueError(f"Unknown role: {role}")
        if not conversation:
            raise LLMGenerationError("No user message to send to the LLM.")
        last = conversation[-1]
        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, cls._build_prompt(conversation[:-1], last["content"])

    # ------------------------------------------------------------------
    # structured output
    # ------------------------------------------------------------------

    @staticmethod
    def _json_schema(
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
    ) -> Optional[dict[str, Any]]:
        """Builds the JSON schema handed to the CLI.

        Both CLIs are backed by APIs whose strict JSON-schema mode requires
        closed objects (``additionalProperties: false``, every property
        required) and rejects open-ended maps, so the schema goes through the
        same transformation as :class:`BaseAnthropicLLM`; it is undone in
        :meth:`_postprocess_content`.
        """
        if response_format is None:
            return None
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return _to_anthropic_schema(response_format.model_json_schema())
        return response_format

    # ------------------------------------------------------------------
    # output handling
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> str:
        """Best-effort extraction of a JSON payload from a CLI answer.

        Both CLIs accept a JSON schema, but they may still wrap the payload in a
        code fence or a sentence, which the calling component would reject.
        """
        stripped = text.strip()
        if not stripped:
            return text
        fenced = _JSON_FENCE.search(stripped)
        if fenced:
            stripped = fenced.group(1).strip()
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass
        for opening, closing in (("{", "}"), ("[", "]")):
            start = stripped.find(opening)
            end = stripped.rfind(closing)
            if start != -1 and end > start:
                candidate = stripped[start : end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
        return text

    def _postprocess_content(
        self,
        text: str,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
    ) -> str:
        if response_format is None:
            return text
        # Undo the open-map transformation applied by _json_schema, so the
        # content stays byte-compatible with the caller's Pydantic model.
        return BaseAnthropicLLM._restore_structured_output(
            self._extract_json(text), response_format
        )

    def _auth_error_hint(self, message: str) -> Optional[str]:
        """Returns a helpful message when the CLI reports a login problem."""
        lowered = message.lower()
        markers = ("not logged in", "/login", "please log in", "unauthorized", "401")
        if any(marker in lowered for marker in markers):
            return (
                f"{self.display_name} is not authenticated: {message.strip()}. Run "
                f"`{self.login_command}` once in a terminal to create the OAuth session "
                "this provider reuses."
            )
        return None

    def _fail(self, message: str) -> LLMGenerationError:
        hint = self._auth_error_hint(message)
        return LLMGenerationError(
            hint or f"{self.display_name} failed: {message.strip()}"
        )
