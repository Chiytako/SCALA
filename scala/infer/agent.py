"""Agent-harness adapters for SCALA's openai-harmony chat format
(llm-jp-tokenizer v4): stop conditions (``<|call|>`` and ``<|return|>``), a
tolerant parser turning generated text into channel segments (analysis /
commentary / final) with tool calls, and ``append_tool_result`` to feed tool
replies back into the message list."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["Segment", "ToolCall", "parse_harmony", "stop_token_ids",
           "STOP_TOKENS", "append_tool_result"]

#: ``<|call|>`` ends a tool call (control returns to the harness);
#: ``<|return|>`` ends the assistant's turn.  Generation must halt on either.
STOP_TOKENS = ("<|call|>", "<|return|>")

_SEG = re.compile(
    r"<\|start\|>(?P<role>[^<|]*?)"
    r"(?:\s+to=(?P<to>[^\s<|]+))?"
    r"(?:<\|channel\|>(?P<channel>[^<|]*?))?"
    r"(?:\s*to=(?P<to2>[^\s<|]+))?"
    r"(?:\s*<\|constrain\|>(?P<constrain>[^<|]*?))?"
    r"<\|message\|>(?P<content>.*?)"
    r"(?P<end><\|end\|>|<\|call\|>|<\|return\|>|$)",
    re.DOTALL,
)


@dataclass
class ToolCall:
    name: str
    arguments: Any
    #: raw argument string, kept for arguments that are not valid JSON
    raw: str = ""
    parse_error: Optional[str] = None


@dataclass
class Segment:
    role: str
    channel: str
    content: str
    #: "<|end|>", "<|call|>", "<|return|>", or "" when generation was truncated
    terminator: str = ""
    recipient: Optional[str] = None
    tool_call: Optional[ToolCall] = None


@dataclass
class ParsedTurn:
    segments: list[Segment] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s.tool_call for s in self.segments if s.tool_call]

    @property
    def final(self) -> str:
        """The user-visible answer, or "" if the turn ended in a tool call."""
        for s in reversed(self.segments):
            if s.channel == "final":
                return s.content
        return ""

    @property
    def reasoning(self) -> str:
        return "\n".join(s.content for s in self.segments
                         if s.channel == "analysis")

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls) and self.segments[-1].terminator == "<|call|>"


def _strip_namespace(name: str) -> str:
    """``functions.get_weather`` -> ``get_weather``."""
    return name.split(".", 1)[1] if "." in name else name


def parse_harmony(text: str, role: str = "assistant") -> ParsedTurn:
    """Parse generated harmony text into segments and tool calls.

    Tolerant: unparseable arguments set ``parse_error`` instead of raising.
    A completion begins mid-segment (the generation prompt already ends with
    ``<|start|>assistant``), so the opener is supplied when missing; ``role``
    says whose turn it is.
    """
    stripped = text.lstrip()
    if stripped and not stripped.startswith("<|start|>"):
        text = f"<|start|>{role}" + text

    turn = ParsedTurn()
    for m in _SEG.finditer(text):
        role = (m.group("role") or "").strip()
        channel = (m.group("channel") or "").strip()
        recipient = m.group("to") or m.group("to2")
        content = m.group("content") or ""
        seg = Segment(role=role, channel=channel, content=content,
                      terminator=m.group("end") or "",
                      recipient=recipient)
        # a tool call = segment with a recipient closed by <|call|>; the same
        # channel also carries non-call commentary, hence the strict terminator
        if recipient and seg.terminator == "<|call|>":
            name = _strip_namespace(recipient)
            try:
                args = json.loads(content)
                seg.tool_call = ToolCall(name, args, raw=content)
            except json.JSONDecodeError as e:
                seg.tool_call = ToolCall(name, None, raw=content,
                                         parse_error=str(e))
        turn.segments.append(seg)
    return turn


def stop_token_ids(tok) -> list[int]:
    """Token ids generation must stop on, for this tokenizer."""
    ids = []
    for t in STOP_TOKENS:
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i >= 0:
            ids.append(i)
    if tok.eos_token_id is not None and tok.eos_token_id not in ids:
        ids.append(tok.eos_token_id)
    return ids


def append_tool_result(messages: list[dict], call: ToolCall,
                       result: Any) -> list[dict]:
    """Append the assistant's call and the tool's reply to a message list.
    Both halves are appended so each tool reply is paired with its call."""
    out = list(messages)
    out.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {"name": call.name,
                         "arguments": call.raw or json.dumps(
                             call.arguments, ensure_ascii=False)},
        }],
    })
    out.append({
        "role": "tool",
        "name": call.name,
        "content": result if isinstance(result, str)
        else json.dumps(result, ensure_ascii=False),
    })
    return out
