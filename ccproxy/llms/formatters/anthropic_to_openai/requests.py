"""Request conversion entry points for Anthropic→OpenAI adapters."""

from __future__ import annotations

import json
from typing import Any

from ccproxy.llms.formatters.context import register_request, register_request_tools
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


def _dump_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json", exclude_none=True)
        except TypeError:
            dumped = value.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    return None


def _stringify_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        fallback_parts: list[Any] = []
        for part in content:
            mapping = _dump_mapping(part)
            if mapping is not None:
                if mapping.get("type") == "text" and isinstance(mapping.get("text"), str):
                    text_parts.append(mapping["text"])
                else:
                    fallback_parts.append(mapping)
                continue
            fallback_parts.append(part)

        if text_parts:
            return "\n\n".join(part for part in text_parts if part)
        if fallback_parts:
            try:
                return json.dumps(fallback_parts)
            except TypeError:
                return str(fallback_parts)
        return ""

    if isinstance(content, dict):
        try:
            return json.dumps(content)
        except TypeError:
            return str(content)

    return "" if content is None else str(content)


def _flush_responses_message(
    input_items: list[dict[str, Any]],
    role: str,
    content_parts: list[dict[str, Any]],
) -> None:
    if not content_parts:
        return
    input_items.append(
        {
            "type": "message",
            "role": role,
            "content": list(content_parts),
        }
    )
    content_parts.clear()


def _normalize_function_call_item_id(call_id: Any, fallback_index: int) -> str:
    if isinstance(call_id, str) and call_id:
        candidate = call_id
    else:
        candidate = f"call_{fallback_index}"
    return candidate if candidate.startswith("fc") else f"fc_{candidate}"


def _convert_anthropic_tools_to_openai_functions(
    tools: list[Any] | None,
) -> list[dict[str, Any]]:
    converted_tools: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, anthropic_models.ToolBase):
            continue

        converted_tools.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
        )

    return converted_tools


def _build_responses_payload_from_anthropic_request(
    request: anthropic_models.CreateMessageRequest,
) -> tuple[dict[str, Any], str | None]:
    """Project an Anthropic message request into Responses payload fields."""

    payload_data: dict[str, Any] = {"model": request.model}
    instructions_text: str | None = None

    if request.max_tokens is not None:
        payload_data["max_output_tokens"] = int(request.max_tokens)
    if request.stream:
        payload_data["stream"] = True

    if request.service_tier is not None:
        payload_data["service_tier"] = request.service_tier
    if request.temperature is not None:
        payload_data["temperature"] = request.temperature
    if request.top_p is not None:
        payload_data["top_p"] = request.top_p

    if request.metadata is not None and hasattr(request.metadata, "model_dump"):
        meta_dump = request.metadata.model_dump()
        payload_data["metadata"] = meta_dump

    if request.system:
        if isinstance(request.system, str):
            instructions_text = request.system
            payload_data["instructions"] = request.system
        else:
            joined = "".join(block.text for block in request.system if block.text)
            instructions_text = joined or None
            if joined:
                payload_data["instructions"] = joined

    input_items: list[dict[str, Any]] = []
    for msg in request.messages or []:
        role = msg.role
        content = msg.content
        text_part_type = "input_text" if role == "user" else "output_text"
        content_parts: list[dict[str, Any]] = []

        if isinstance(content, str):
            if content:
                content_parts.append({"type": text_part_type, "text": content})
            _flush_responses_message(input_items, role, content_parts)
            continue

        for block in content or []:
            mapping = _dump_mapping(block)
            if mapping is None:
                continue

            block_type = str(mapping.get("type", "")).lower()
            if block_type == "text":
                text_value = mapping.get("text")
                if isinstance(text_value, str) and text_value:
                    content_parts.append({"type": text_part_type, "text": text_value})
                continue

            if block_type in {"thinking", "redacted_thinking"}:
                continue

            if block_type == "image" and role == "user":
                source = mapping.get("source")
                if isinstance(source, dict):
                    media_type = source.get("media_type")
                    data = source.get("data")
                    if (
                        source.get("type") == "base64"
                        and isinstance(media_type, str)
                        and isinstance(data, str)
                    ):
                        content_parts.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:{media_type};base64,{data}",
                            }
                        )
                continue

            if block_type == "tool_use" and role == "assistant":
                _flush_responses_message(input_items, role, content_parts)
                tool_input = mapping.get("input")
                call_id = mapping.get("id") or ""
                if isinstance(tool_input, str):
                    arguments = tool_input
                elif tool_input is None:
                    arguments = "{}"
                else:
                    try:
                        arguments = json.dumps(tool_input)
                    except TypeError:
                        arguments = json.dumps({"arguments": str(tool_input)})
                input_items.append(
                    {
                        "type": "function_call",
                        "id": _normalize_function_call_item_id(
                            call_id, len(input_items)
                        ),
                        "call_id": call_id,
                        "name": mapping.get("name") or "",
                        "arguments": arguments,
                    }
                )
                continue

            if block_type == "tool_result":
                _flush_responses_message(input_items, role, content_parts)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": mapping.get("tool_use_id") or "",
                        "output": _stringify_tool_result_content(
                            mapping.get("content")
                        ),
                    }
                )

        _flush_responses_message(input_items, role, content_parts)

    payload_data["input"] = input_items

    tools = _convert_anthropic_tools_to_openai_functions(request.tools)
    if tools:
        payload_data["tools"] = tools

    tc = request.tool_choice
    if tc is not None:
        tc_type = getattr(tc, "type", None)
        if tc_type == "none":
            payload_data["tool_choice"] = "none"
        elif tc_type == "auto":
            payload_data["tool_choice"] = "auto"
        elif tc_type == "any":
            payload_data["tool_choice"] = "required"
        elif tc_type == "tool":
            name = getattr(tc, "name", None)
            if name:
                payload_data["tool_choice"] = {
                    "type": "function",
                    "name": name,
                }
        disable_parallel = getattr(tc, "disable_parallel_tool_use", None)
        if isinstance(disable_parallel, bool):
            payload_data["parallel_tool_calls"] = not disable_parallel

    payload_data.setdefault("background", None)

    return payload_data, instructions_text


def convert__anthropic_message_to_openai_responses__request(
    request: anthropic_models.CreateMessageRequest,
) -> openai_models.ResponseRequest:
    """Convert Anthropic CreateMessageRequest to OpenAI ResponseRequest using typed models."""
    payload_data, instructions_text = _build_responses_payload_from_anthropic_request(
        request
    )

    response_request = openai_models.ResponseRequest.model_validate(payload_data)

    register_request_tools(request.tools)
    register_request(request, instructions_text)

    return response_request


def convert__anthropic_message_to_openai_chat__request(
    request: anthropic_models.CreateMessageRequest,
) -> openai_models.ChatCompletionRequest:
    """Convert Anthropic CreateMessageRequest to OpenAI ChatCompletionRequest using typed models."""
    openai_messages: list[dict[str, Any]] = []
    # System prompt
    if request.system:
        if isinstance(request.system, str):
            sys_content = request.system
        else:
            sys_content = "".join(block.text for block in request.system)
        if sys_content:
            openai_messages.append({"role": "system", "content": sys_content})

    # User/assistant messages with text + data-url images
    for msg in request.messages:
        role = msg.role
        content = msg.content

        # Handle tool usage and results
        if role == "assistant" and isinstance(content, list):
            tool_calls = []
            text_parts = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "tool_use":
                    # Type guard for ToolUseBlock
                    if hasattr(block, "id") and hasattr(block, "name"):
                        # Safely get input with fallback to empty dict
                        tool_input = getattr(block, "input", {}) or {}

                        # Ensure input is properly serialized as JSON
                        try:
                            args_str = json.dumps(tool_input)
                        except Exception:
                            args_str = json.dumps({"arguments": str(tool_input)})

                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": args_str,
                                },
                            }
                        )
                elif block_type == "text":
                    # Type guard for TextBlock
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
            if tool_calls:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }
                assistant_msg["content"] = " ".join(text_parts) if text_parts else None
                openai_messages.append(assistant_msg)
                continue
        elif role == "user" and isinstance(content, list):
            is_tool_result = any(
                getattr(b, "type", None) == "tool_result" for b in content
            )
            if is_tool_result:
                for block in content:
                    if getattr(block, "type", None) == "tool_result":
                        # Type guard for ToolResultBlock
                        if hasattr(block, "tool_use_id"):
                            # Get content with an empty string fallback
                            result_content = getattr(block, "content", "")

                            # Convert complex content to string representation
                            if not isinstance(result_content, str):
                                try:
                                    if isinstance(result_content, list):
                                        # Handle list of text blocks
                                        text_parts = []
                                        for part in result_content:
                                            if (
                                                hasattr(part, "text")
                                                and hasattr(part, "type")
                                                and part.type == "text"
                                            ):
                                                text_parts.append(part.text)
                                        if text_parts:
                                            result_content = " ".join(text_parts)
                                        else:
                                            result_content = json.dumps(result_content)
                                    else:
                                        # Convert other non-string content to JSON
                                        result_content = json.dumps(result_content)
                                except Exception:
                                    # Fallback to string representation
                                    result_content = str(result_content)

                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.tool_use_id,
                                    "content": result_content,
                                }
                            )
                continue

        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            text_accum: list[str] = []
            for block in content:
                # Support both raw dicts and Anthropic model instances
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text" and isinstance(block.get("text"), str):
                        text_accum.append(block.get("text") or "")
                    elif btype == "image":
                        source = block.get("source") or {}
                        if (
                            isinstance(source, dict)
                            and source.get("type") == "base64"
                            and isinstance(source.get("media_type"), str)
                            and isinstance(source.get("data"), str)
                        ):
                            url = f"data:{source['media_type']};base64,{source['data']}"
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url},
                                }
                            )
                else:
                    # Pydantic models
                    btype = getattr(block, "type", None)
                    if (
                        btype == "text"
                        and hasattr(block, "text")
                        and isinstance(getattr(block, "text", None), str)
                    ):
                        text_accum.append(block.text or "")
                    elif btype == "image":
                        source = getattr(block, "source", None)
                        if (
                            source is not None
                            and getattr(source, "type", None) == "base64"
                            and isinstance(getattr(source, "media_type", None), str)
                            and isinstance(getattr(source, "data", None), str)
                        ):
                            url = f"data:{source.media_type};base64,{source.data}"
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url},
                                }
                            )
            if parts or len(text_accum) > 1:
                if text_accum:
                    parts.insert(0, {"type": "text", "text": " ".join(text_accum)})
                openai_messages.append({"role": role, "content": parts})
            else:
                openai_messages.append(
                    {"role": role, "content": (text_accum[0] if text_accum else "")}
                )
        else:
            openai_messages.append({"role": role, "content": content})

    # Tools mapping (custom tools -> function tools)
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
            },
        }
        for tool in _convert_anthropic_tools_to_openai_functions(request.tools)
    ]

    params: dict[str, Any] = {
        "model": request.model,
        "messages": openai_messages,
        "max_completion_tokens": request.max_tokens,
        "stream": request.stream or None,
    }
    if tools:
        params["tools"] = tools

    # tool_choice mapping
    tc = request.tool_choice
    if tc is not None:
        tc_type = getattr(tc, "type", None)
        if tc_type == "none":
            params["tool_choice"] = "none"
        elif tc_type == "auto":
            params["tool_choice"] = "auto"
        elif tc_type == "any":
            params["tool_choice"] = "required"
        elif tc_type == "tool":
            name = getattr(tc, "name", None)
            if name:
                params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": name},
                }
        # parallel_tool_calls from disable_parallel_tool_use
        disable_parallel = getattr(tc, "disable_parallel_tool_use", None)
        if isinstance(disable_parallel, bool):
            params["parallel_tool_calls"] = not disable_parallel

    register_request_tools(request.tools)

    # Validate against OpenAI model
    return openai_models.ChatCompletionRequest.model_validate(params)


__all__ = [
    "convert__anthropic_message_to_openai_chat__request",
    "convert__anthropic_message_to_openai_responses__request",
]
