import os
from typing import Any, Callable, Optional

import anthropic

from tools import TOOL_DEFINITIONS, run_tool

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = """You are a concise shopping assistant in a terminal chat.

When the user asks for a total cost, call the calculate_total tool. Do not do \
the arithmetic yourself. Pass tax_rate as a decimal fraction (10% -> 0.10). If \
a follow-up question changes only some values (for example the quantity), \
reuse the other values from earlier in the conversation. If a required value \
was never given, ask the user for it instead of guessing.

If a tool returns an error, read it, correct the arguments, and try again, or \
explain the problem to the user.

Answer in one or two short sentences of plain text, formatting money like \
$2,805 or $2,805.50."""

ToolCallback = Callable[[str, Any, str, bool], None]


class AssistantError(Exception):
    pass


class Assistant:
    def __init__(
        self,
        client: Optional[Any] = None,
        model: Optional[str] = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        on_tool_call: Optional[ToolCallback] = None,
    ):
        self.client = client if client is not None else anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.max_tool_rounds = max_tool_rounds
        self.on_tool_call = on_tool_call
        self.messages: list[dict] = []

    def ask(self, user_text: str) -> str:
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        try:
            return self._run_turn()
        except Exception:
            del self.messages[checkpoint:]
            raise

    def _run_turn(self) -> str:
        for _ in range(self.max_tool_rounds + 1):
            response = self._call_llm()

            if response.stop_reason == "refusal":
                raise AssistantError("The model declined to answer that request.")
            if response.stop_reason == "max_tokens":
                raise AssistantError("The response was cut off (max_tokens reached).")

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return _text_of(response)

            tool_results = [
                self._execute(block) for block in response.content if block.type == "tool_use"
            ]
            self.messages.append({"role": "user", "content": tool_results})

        raise AssistantError(
            f"Stopped after {self.max_tool_rounds} rounds of tool calls without a final answer."
        )

    def _call_llm(self):
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=self.messages,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def _execute(self, block) -> dict:
        content, is_error = run_tool(block.name, block.input)
        if self.on_tool_call:
            self.on_tool_call(block.name, block.input, content, is_error)
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": content,
            "is_error": is_error,
        }


def _text_of(response) -> str:
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or "(no response)"
