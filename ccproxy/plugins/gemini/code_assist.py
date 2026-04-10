"""Helpers for translating between OpenAI chat payloads and Gemini Code Assist."""

from __future__ import annotations

import json
import platform
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping
from typing import Any

from ccproxy.llms.formatters.openai_to_anthropic.streams import (
    OpenAIChatToAnthropicStreamAdapter,
)
from ccproxy.llms.formatters.openai_to_openai.streams import (
    convert__openai_chat_to_openai_responses__stream,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


def build_client_metadata() -> dict[str, str]:
    """Build minimal Gemini Code Assist client metadata."""

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin" and machine in {"x86_64", "amd64"}:
        platform_name = "DARWIN_AMD64"
    elif system == "darwin" and machine in {"arm64", "aarch64"}:
        platform_name = "DARWIN_ARM64"
    elif system == "linux" and machine in {"x86_64", "amd64"}:
        platform_name = "LINUX_AMD64"
    elif system == "linux" and machine in {"arm64", "aarch64"}:
        platform_name = "LINUX_ARM64"
    elif system == "windows" and machine in {"x86_64", "amd64"}:
        platform_name = "WINDOWS_AMD64"
    else:
        platform_name = "PLATFORM_UNSPECIFIED"

    return {
        "ideName": "IDE_UNSPECIFIED",
        "pluginType": "GEMINI",
        "platform": platform_name,
    }


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return None


def _ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _content_parts_from_openai(content: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []

    def add_text(value: Any) -> None:
        text = _ensure_text(value)
        if text:
            parts.append({"text": text})

    if isinstance(content, str):
        add_text(content)
        return parts

    if not isinstance(content, list):
        return parts

    for item in content:
        if isinstance(item, str):
            add_text(item)
            continue
        if not isinstance(item, Mapping):
            add_text(item)
            continue

        item_type = item.get("type")
        if item_type in {
            "text",
            "input_text",
            "output_text",
            "text_delta",
            "thinking",
            "thinking_delta",
        }:
            add_text(item.get("text") or item.get("thinking"))
            continue

        if item_type == "tool_result":
            add_text(item.get("content"))
            continue

        if "text" in item:
            add_text(item.get("text"))

    return parts


def _function_declarations_from_tools(tools: Any) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return declarations

    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        if tool.get("type") != "function":
            continue

        function_data = tool.get("function")
        if isinstance(function_data, Mapping):
            name = function_data.get("name")
            description = function_data.get("description")
            parameters = function_data.get("parameters") or {}
        else:
            name = tool.get("name")
            description = tool.get("description")
            parameters = tool.get("parameters") or {}

        if not isinstance(name, str) or not name:
            continue

        declarations.append(
            {
                "name": name,
                "description": _ensure_text(description) if description else "",
                "parameters": parameters if isinstance(parameters, Mapping) else {},
            }
        )

    return declarations


def _tool_config_from_choice(tool_choice: Any) -> dict[str, Any] | None:
    if tool_choice in (None, "auto"):
        return None
    if tool_choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if tool_choice == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}

    if isinstance(tool_choice, Mapping):
        choice_type = tool_choice.get("type")
        if choice_type == "function":
            function_data = tool_choice.get("function")
            if isinstance(function_data, Mapping):
                name = function_data.get("name")
                if isinstance(name, str) and name:
                    return {
                        "functionCallingConfig": {
                            "mode": "ANY",
                            "allowedFunctionNames": [name],
                        }
                    }
        if choice_type == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if choice_type in {"required", "any", "tool"}:
            allowed_name = tool_choice.get("name")
            config: dict[str, Any] = {"mode": "ANY"}
            if isinstance(allowed_name, str) and allowed_name:
                config["allowedFunctionNames"] = [allowed_name]
            return {"functionCallingConfig": config}

    return None


def _generation_config_from_openai(payload: Mapping[str, Any]) -> dict[str, Any]:
    generation_config: dict[str, Any] = {}

    if isinstance(payload.get("temperature"), (int, float)):
        generation_config["temperature"] = payload["temperature"]
    if isinstance(payload.get("top_p"), (int, float)):
        generation_config["topP"] = payload["top_p"]
    if isinstance(payload.get("max_completion_tokens"), int):
        generation_config["maxOutputTokens"] = payload["max_completion_tokens"]
    elif isinstance(payload.get("max_tokens"), int):
        generation_config["maxOutputTokens"] = payload["max_tokens"]

    stop = payload.get("stop")
    if isinstance(stop, str):
        generation_config["stopSequences"] = [stop]
    elif isinstance(stop, list):
        generation_config["stopSequences"] = [
            str(item) for item in stop if isinstance(item, str) and item
        ]

    response_format = payload.get("response_format")
    if isinstance(response_format, Mapping):
        fmt_type = response_format.get("type")
        if fmt_type in {"json_object", "json_schema"}:
            generation_config["responseMimeType"] = "application/json"
            schema = response_format.get("json_schema")
            if isinstance(schema, Mapping):
                generation_config["responseSchema"] = dict(schema)

    return generation_config


def openai_chat_to_code_assist_request(
    payload: Mapping[str, Any],
    *,
    project_id: str | None,
    session_id: str | None = None,
    user_prompt_id: str | None = None,
    thought_signatures: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Translate an OpenAI chat-completions request into Code Assist format."""

    messages = payload.get("messages")
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, Any]] = []
    tool_names_by_call_id: dict[str, str] = {}
    resolved_signatures = dict(thought_signatures or {})

    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping):
                continue

            role = str(message.get("role") or "")
            content_parts = _content_parts_from_openai(message.get("content"))

            if role in {"system", "developer"}:
                system_parts.extend(content_parts)
                continue

            if role == "assistant":
                parts = list(content_parts)
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for index, tool_call in enumerate(tool_calls):
                        if not isinstance(tool_call, Mapping):
                            continue
                        function_data = tool_call.get("function")
                        if not isinstance(function_data, Mapping):
                            continue
                        name = function_data.get("name")
                        if not isinstance(name, str) or not name:
                            continue
                        raw_arguments = function_data.get("arguments")
                        parsed_arguments = (
                            _safe_json_loads(raw_arguments)
                            if isinstance(raw_arguments, str)
                            else raw_arguments
                        )
                        if not isinstance(parsed_arguments, Mapping):
                            parsed_arguments = {
                                "raw_arguments": _ensure_text(raw_arguments)
                            }
                        tool_call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex}_{index}"
                        tool_names_by_call_id[str(tool_call_id)] = name
                        function_call_payload = {
                            "id": str(tool_call_id),
                            "name": name,
                            "args": dict(parsed_arguments),
                        }
                        thought_signature = resolved_signatures.get(str(tool_call_id))
                        if isinstance(thought_signature, str) and thought_signature:
                            function_call_payload["thoughtSignature"] = thought_signature
                        parts.append({"functionCall": function_call_payload})
                if parts:
                    contents.append({"role": "model", "parts": parts})
                continue

            if role == "tool":
                tool_call_id = message.get("tool_call_id")
                tool_name = message.get("name") or tool_names_by_call_id.get(
                    str(tool_call_id), "tool"
                )
                response_payload: Any
                if isinstance(message.get("content"), str):
                    response_payload = {"output": message["content"]}
                elif content_parts:
                    response_payload = {"output": "\n".join(
                        part["text"] for part in content_parts if part.get("text")
                    )}
                else:
                    response_payload = {"output": _ensure_text(message.get("content"))}

                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "id": str(tool_call_id) if tool_call_id else None,
                                    "name": str(tool_name),
                                    "response": response_payload,
                                }
                            }
                        ],
                    }
                )
                continue

            if role == "user" and content_parts:
                contents.append({"role": "user", "parts": content_parts})

    request_payload: dict[str, Any] = {
        "model": str(payload.get("model") or ""),
        "user_prompt_id": user_prompt_id or f"ccproxy-{uuid.uuid4().hex}",
        "request": {
            "contents": contents,
        },
    }
    if project_id:
        request_payload["project"] = project_id
    if system_parts:
        request_payload["request"]["systemInstruction"] = {
            "role": "user",
            "parts": system_parts,
        }
    if session_id:
        request_payload["request"]["session_id"] = session_id

    tools = _function_declarations_from_tools(payload.get("tools"))
    if tools:
        request_payload["request"]["tools"] = [{"functionDeclarations": tools}]

    tool_config = _tool_config_from_choice(payload.get("tool_choice"))
    if tool_config:
        request_payload["request"]["toolConfig"] = tool_config

    generation_config = _generation_config_from_openai(payload)
    if generation_config:
        request_payload["request"]["generationConfig"] = generation_config

    return request_payload


def openai_chat_to_code_assist_count_request(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a Code Assist countTokens request from an OpenAI chat payload."""

    request_payload = openai_chat_to_code_assist_request(payload, project_id=None)
    request_body = request_payload.get("request", {})
    contents = request_body.get("contents")
    if not isinstance(contents, list):
        contents = []

    system_instruction = request_body.get("systemInstruction")
    if isinstance(system_instruction, Mapping) and system_instruction.get("parts"):
        contents = [dict(system_instruction), *contents]

    return {
        "request": {
            "model": f"models/{payload.get('model')}",
            "contents": contents,
        }
    }


def _build_usage(usage_metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(usage_metadata, Mapping):
        return None

    prompt_tokens = int(usage_metadata.get("promptTokenCount") or 0)
    completion_tokens = int(usage_metadata.get("candidatesTokenCount") or 0)
    total_tokens = int(
        usage_metadata.get("totalTokenCount") or prompt_tokens + completion_tokens
    )
    reasoning_tokens = int(usage_metadata.get("thoughtsTokenCount") or 0)

    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    usage["completion_tokens_details"] = {
        "reasoning_tokens": reasoning_tokens,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    }
    usage["prompt_tokens_details"] = {
        "cached_tokens": 0,
        "audio_tokens": 0,
    }
    return usage


def _tool_call_id(base_id: str, index: int, raw_id: Any = None) -> str:
    if isinstance(raw_id, str) and raw_id:
        return raw_id
    return f"call_{base_id}_{index}"


def collect_tool_signatures_from_code_assist_payload(
    payload: Mapping[str, Any],
) -> dict[str, str]:
    """Collect thoughtSignature values keyed by tool-call id."""

    response_payload = payload.get("response")
    if not isinstance(response_payload, Mapping):
        return {}

    candidates = response_payload.get("candidates")
    candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
    if not isinstance(candidate, Mapping):
        return {}

    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list):
        return {}

    response_id = str(
        payload.get("traceId")
        or response_payload.get("responseId")
        or "toolcall"
    )
    collected: dict[str, str] = {}
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            continue
        function_call = part.get("functionCall")
        if not isinstance(function_call, Mapping):
            continue
        thought_signature = function_call.get("thoughtSignature")
        if not isinstance(thought_signature, str) or not thought_signature:
            continue
        tool_call_id = _tool_call_id(response_id, index, function_call.get("id"))
        collected[tool_call_id] = thought_signature

    return collected


def code_assist_to_openai_chat_response(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate a Code Assist response into an OpenAI chat-completions response."""

    response_payload = payload.get("response")
    if not isinstance(response_payload, Mapping):
        return {
            "id": payload.get("traceId") or f"chatcmpl_{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": ""},
                }
            ],
        }

    candidates = response_payload.get("candidates")
    candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
    if not isinstance(candidate, Mapping):
        candidate = {}

    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list):
        parts = []

    created = int(time.time())
    model_id = str(
        response_payload.get("modelVersion")
        or payload.get("modelVersion")
        or ""
    )
    response_id = str(
        payload.get("traceId")
        or response_payload.get("responseId")
        or f"chatcmpl_{uuid.uuid4().hex}"
    )

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)

        function_call = part.get("functionCall")
        if isinstance(function_call, Mapping):
            name = function_call.get("name")
            if not isinstance(name, str) or not name:
                continue
            arguments = function_call.get("args")
            tool_calls.append(
                {
                    "id": _tool_call_id(response_id, index, function_call.get("id")),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments or {}, ensure_ascii=False),
                    },
                }
            )

    finish_reason_raw = str(candidate.get("finishReason") or "").upper()
    if tool_calls:
        finish_reason = "tool_calls"
    elif finish_reason_raw == "MAX_TOKENS":
        finish_reason = "length"
    else:
        finish_reason = "stop"

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = _build_usage(response_payload.get("usageMetadata"))

    response: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": message,
            }
        ],
    }
    if usage is not None:
        response["usage"] = usage
    return response


async def code_assist_stream_to_openai_chat_chunks(
    events: AsyncIterator[Mapping[str, Any]],
) -> AsyncGenerator[openai_models.ChatCompletionChunk, None]:
    """Translate Code Assist SSE payloads into OpenAI chat-completion chunks."""

    started = False
    emitted_final = False
    latest_id = f"chatcmpl_{uuid.uuid4().hex}"
    latest_model = ""
    created = int(time.time())

    async for payload in events:
        response_payload = payload.get("response")
        if not isinstance(response_payload, Mapping):
            continue

        latest_id = str(
            payload.get("traceId")
            or response_payload.get("responseId")
            or latest_id
        )
        latest_model = str(response_payload.get("modelVersion") or latest_model)
        candidates = response_payload.get("candidates")
        candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
        if not isinstance(candidate, Mapping):
            candidate = {}

        if not started:
            started = True
            yield openai_models.ChatCompletionChunk(
                id=latest_id,
                created=created,
                model=latest_model or None,
                choices=[
                    openai_models.StreamingChoice(
                        index=0,
                        delta=openai_models.DeltaMessage(role="assistant"),
                        finish_reason=None,
                    )
                ],
            )

        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, Mapping) else None
        if not isinstance(parts, list):
            parts = []

        for index, part in enumerate(parts):
            if not isinstance(part, Mapping):
                continue

            text = part.get("text")
            if isinstance(text, str) and text:
                yield openai_models.ChatCompletionChunk(
                    id=latest_id,
                    created=created,
                    model=latest_model or None,
                    choices=[
                        openai_models.StreamingChoice(
                            index=0,
                            delta=openai_models.DeltaMessage(content=text),
                            finish_reason=None,
                        )
                    ],
                )

            function_call = part.get("functionCall")
            if isinstance(function_call, Mapping):
                name = function_call.get("name")
                if not isinstance(name, str) or not name:
                    continue
                arguments = json.dumps(
                    function_call.get("args") or {},
                    ensure_ascii=False,
                )
                yield openai_models.ChatCompletionChunk(
                    id=latest_id,
                    created=created,
                    model=latest_model or None,
                    choices=[
                        openai_models.StreamingChoice(
                            index=0,
                            delta=openai_models.DeltaMessage(
                                tool_calls=[
                                    openai_models.ToolCallChunk(
                                        index=index,
                                        id=_tool_call_id(
                                            latest_id,
                                            index,
                                            function_call.get("id"),
                                        ),
                                        type="function",
                                        function=openai_models.FunctionCall(
                                            name=name,
                                            arguments=arguments,
                                        ),
                                    )
                                ]
                            ),
                            finish_reason=None,
                        )
                    ],
                )

        finish_reason_raw = str(candidate.get("finishReason") or "").upper()
        has_tool_calls = any(
            isinstance(part, Mapping) and isinstance(part.get("functionCall"), Mapping)
            for part in parts
        )
        if finish_reason_raw or response_payload.get("usageMetadata"):
            emitted_final = True
            if has_tool_calls:
                finish_reason = "tool_calls"
            elif finish_reason_raw == "MAX_TOKENS":
                finish_reason = "length"
            else:
                finish_reason = "stop"

            yield openai_models.ChatCompletionChunk(
                id=latest_id,
                created=created,
                model=latest_model or None,
                choices=[
                    openai_models.StreamingChoice(
                        index=0,
                        delta=openai_models.DeltaMessage(),
                        finish_reason=finish_reason,
                    )
                ],
                usage=(
                    openai_models.CompletionUsage.model_validate(usage)
                    if (usage := _build_usage(response_payload.get("usageMetadata")))
                    else None
                ),
            )

    if not emitted_final:
        yield openai_models.ChatCompletionChunk(
            id=latest_id,
            created=created,
            model=latest_model or None,
            choices=[
                openai_models.StreamingChoice(
                    index=0,
                    delta=openai_models.DeltaMessage(),
                    finish_reason="stop",
                )
            ],
        )


async def openai_chat_chunks_to_anthropic_events(
    stream: AsyncIterator[openai_models.ChatCompletionChunk],
) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
    adapter = OpenAIChatToAnthropicStreamAdapter()
    async for event in adapter.run(stream):
        yield event


async def openai_chat_chunks_to_responses_events(
    stream: AsyncIterator[openai_models.ChatCompletionChunk],
) -> AsyncGenerator[dict[str, Any], None]:
    async for event in convert__openai_chat_to_openai_responses__stream(stream):
        if hasattr(event, "model_dump"):
            yield event.model_dump(mode="json", exclude_none=True)
        elif isinstance(event, Mapping):
            yield dict(event)

