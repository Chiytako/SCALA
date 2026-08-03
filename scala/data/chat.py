"""Render conversation-shaped records into one harmony-formatted string.

Records are normalised to the OpenAI message shape first (ShareGPT from/value,
varying reasoning fields, tools as list or JSON string all handled), then
rendered with the tokenizer's own chat template -- the student must emit the
exact byte sequence an agent harness parses back (llm-jp-tokenizer v4 harmony).
"""

from __future__ import annotations

import json
from typing import Any, Iterable

__all__ = ["normalise_messages", "render_chat", "ChatRenderError"]


class ChatRenderError(ValueError):
    """A record could not be turned into a training string."""


#: ShareGPT-style speaker labels -> OpenAI roles.  `gpt`/`chatgpt`/`bard` all
#: appear in the wild for the assistant turn.
_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bard": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
    "function_call": "assistant",
}


def _as_list(v: Any) -> list:
    """Tool specs arrive either as a list or as a JSON string of a list."""
    if v is None or v == "":
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def normalise_messages(raw: Iterable[dict]) -> list[dict]:
    """Coerce any observed message shape into OpenAI messages.

    Reasoning text (``reasoning``, ``reasoning_content``, or inline
    ``<think>`` tags) is folded into ``content``, not dropped.
    """
    out: list[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        # ShareGPT: {"from": "human", "value": "..."}
        if "from" in m and "value" in m:
            role = _ROLE_MAP.get(str(m["from"]).lower())
            if role is None:
                continue
            out.append({"role": role, "content": m["value"] or ""})
            continue

        role = _ROLE_MAP.get(str(m.get("role", "")).lower())
        if role is None:
            continue
        content = m.get("content") or ""
        if isinstance(content, list):
            # multimodal-style content blocks; keep the text parts
            content = "".join(b.get("text", "") for b in content
                              if isinstance(b, dict))

        reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
        if reasoning and role == "assistant":
            # <think> wrapping keeps the reasoning/answer boundary machine-findable
            content = f"<think>\n{reasoning.strip()}\n</think>\n{content}"

        msg: dict[str, Any] = {"role": role, "content": content}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        if m.get("name"):
            msg["name"] = m["name"]
        out.append(msg)

    # a transcript with no assistant turn is not trainable
    if not any(m["role"] == "assistant" for m in out):
        raise ChatRenderError("no assistant turn")
    return out


def _manual_harmony(messages: list[dict], tools: list) -> str:
    """Fallback when the tokenizer's own template rejects a record.

    Handles tool_calls the strict template will not render (missing ``id``,
    stringified arguments, parallel calls).  Close to the template's plain-case
    output; byte-identity with template edge cases is not a goal.
    """
    parts: list[str] = []
    if tools:
        spec = json.dumps(tools, ensure_ascii=False)
        parts.append(f"<|start|>system<|message|># Tools\n{spec}<|end|>")
    for m in messages:
        role, content = m["role"], m.get("content", "")
        if role == "assistant" and m.get("tool_calls"):
            for tc in _as_list(m["tool_calls"]):
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                name = fn.get("name", "tool")
                args = fn.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                parts.append(
                    f"<|start|>assistant<|channel|>commentary "
                    f"to={name}<|constrain|>json<|message|>{args}<|call|>")
            if content:
                parts.append(f"<|start|>assistant<|channel|>final"
                             f"<|message|>{content}<|end|>")
        elif role == "assistant":
            parts.append(f"<|start|>assistant<|channel|>final"
                         f"<|message|>{content}<|return|>")
        elif role == "tool":
            name = m.get("name", "tool")
            parts.append(f"<|start|>{name} to=assistant<|channel|>commentary"
                         f"<|message|>{content}<|end|>")
        else:
            parts.append(f"<|start|>{role}<|message|>{content}<|end|>")
    return "".join(parts)


def messages_from_fields(rec: dict, spec: dict[str, str]) -> list[dict]:
    """Build a conversation out of flat columns (plain QA tables).

    ``spec`` maps a role to a format string over the record's columns::

        chat_from_fields:
          user:      "{instruction}\\n\\n{question}"
          assistant: "{answer}"
    """
    msgs = []
    for role in ("system", "user", "assistant"):
        tmpl = spec.get(role)
        if not tmpl:
            continue
        try:
            content = tmpl.format(**{k: (v if v is not None else "")
                                     for k, v in rec.items()})
        except KeyError as e:
            raise ChatRenderError(f"chat_from_fields refers to missing "
                                  f"column {e}")
        content = content.strip()
        if content:
            msgs.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" for m in msgs):
        raise ChatRenderError("chat_from_fields produced no assistant turn")
    return msgs


def render_chat(rec: dict, tok, messages_key: str | None = None,
                tools_key: str = "tools",
                from_fields: dict[str, str] | None = None) -> str:
    """Return one training string for a conversation record.

    Raises ``ChatRenderError`` when the record holds nothing trainable;
    callers should skip such records rather than abort.
    """
    if from_fields:
        messages = messages_from_fields(rec, from_fields)
        return _finish(messages, _as_list(rec.get(tools_key)), tok)

    if messages_key:
        raw = rec.get(messages_key)
    else:
        raw = rec.get("messages") or rec.get("conversations")
    if not raw:
        raise ChatRenderError("no messages/conversations column")
    # some corpora store the conversation as a JSON string, not a list of dicts
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ChatRenderError(f"messages is a string but not JSON: {e}")
        if not isinstance(raw, list):
            raise ChatRenderError("messages JSON is not a list")

    return _finish(normalise_messages(raw), _as_list(rec.get(tools_key)), tok)


def _finish(messages: list[dict], tools: list, tok) -> str:
    try:
        text = tok.apply_chat_template(
            messages, tools=tools or None, tokenize=False,
            add_generation_prompt=False)
    except Exception:  # noqa: BLE001 -- template rejection is expected
        text = _manual_harmony(messages, tools)
    if not text or not text.strip():
        raise ChatRenderError("template produced empty text")
    return text
