"""This example demonstrates how to use the agent CLIs already installed on your
machine -- Claude Code and Codex -- as LLM providers.

No API key is involved: each CLI is spawned in non-interactive mode and signs in
with the OAuth session you created with `claude /login` / `codex login`, so a
Claude or ChatGPT subscription works as-is (within its own usage limits and
terms).

    python examples/customize/llms/local_cli_llm.py
    python examples/customize/llms/local_cli_llm.py --cli codex
"""

import argparse

from neo4j_graphrag.llm import ClaudeCodeLLM, CodexCLILLM, LLMResponse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--cli",
    choices=["claude", "codex"],
    default="claude",
    help="which locally installed CLI to use",
)
parser.add_argument(
    "--model",
    default="",
    help="model to ask the CLI for; defaults to the CLI's own configuration",
)
args = parser.parse_args()

llm_class = ClaudeCodeLLM if args.cli == "claude" else CodexCLILLM

# Optional: check the OAuth session before running anything. This only reads
# metadata (never the tokens) and the CLI remains the source of truth.
status = llm_class.auth_status()
print(f"{status.cli}: authenticated={status.authenticated} ({status.detail})")

with llm_class(model_name=args.model) as llm:
    print("executable:", llm.executable)
    res: LLMResponse = llm.invoke("Who is the mother of Paul Atreides?")
    print(res.content)
    print("usage:", res.usage)
