"""API routes for the Gemini provider plugin."""

from __future__ import annotations

import json
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

from ccproxy.api.decorators import with_format_chain
from ccproxy.api.dependencies import (
    get_plugin_adapter,
    get_provider_config_dependency,
)
from ccproxy.auth.dependencies import ConditionalAuthDep
from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
    UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models
from ccproxy.streaming import DeferredStreaming

from .config import GeminiConfig


GeminiAdapterDep = Annotated[Any, Depends(get_plugin_adapter("gemini"))]
GeminiConfigDep = Annotated[
    GeminiConfig,
    Depends(get_provider_config_dependency("gemini", GeminiConfig)),
]

APIResponse = Response | StreamingResponse | DeferredStreaming

router = APIRouter()


def _estimate_count_tokens_payload(
    request_body: anthropic_models.CountMessageTokensRequest,
) -> int:
    payload = request_body.model_dump(mode="json", exclude_none=True)
    payload.pop("model", None)

    messages = payload.get("messages")
    tools = payload.get("tools")
    system = payload.get("system")

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    token_count = (len(serialized.encode("utf-8")) + 3) // 4 if serialized else 0

    if isinstance(messages, list):
        token_count += len(messages) * 8
    if isinstance(tools, list):
        token_count += len(tools) * 32
    if system:
        token_count += 8

    return token_count


def _cast_result(result: object) -> APIResponse:
    return cast(APIResponse, result)


async def _handle_adapter_request(
    request: Request,
    adapter: Any,
) -> APIResponse:
    result = await adapter.handle_request(request)
    return _cast_result(result)


@router.post(
    "/v1/chat/completions",
    response_model=openai_models.ChatCompletionResponse | openai_models.ErrorResponse,
)
@with_format_chain(
    [FORMAT_OPENAI_CHAT],
    endpoint=UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
async def gemini_chat_completions(
    request: Request,
    _: openai_models.ChatCompletionRequest,
    auth: ConditionalAuthDep,
    adapter: GeminiAdapterDep,
) -> APIResponse:
    return await _handle_adapter_request(request, adapter)


@router.post("/v1/responses", response_model=None)
@with_format_chain(
    [FORMAT_OPENAI_RESPONSES, FORMAT_OPENAI_CHAT],
    endpoint=UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
async def gemini_responses(
    request: Request,
    _: openai_models.ResponseRequest,
    auth: ConditionalAuthDep,
    adapter: GeminiAdapterDep,
) -> APIResponse:
    return await _handle_adapter_request(request, adapter)


@router.post(
    "/v1/messages",
    response_model=anthropic_models.MessageResponse | anthropic_models.APIError,
)
@with_format_chain(
    [FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_CHAT],
    endpoint=UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
async def gemini_messages(
    request: Request,
    _: anthropic_models.CreateMessageRequest,
    auth: ConditionalAuthDep,
    adapter: GeminiAdapterDep,
) -> APIResponse:
    return await _handle_adapter_request(request, adapter)


@router.post(
    "/{session_id}/v1/messages",
    response_model=anthropic_models.MessageResponse | anthropic_models.APIError,
)
@with_format_chain(
    [FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_CHAT],
    endpoint=UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
async def gemini_messages_with_session(
    session_id: str,
    request: Request,
    _: anthropic_models.CreateMessageRequest,
    auth: ConditionalAuthDep,
    adapter: GeminiAdapterDep,
) -> APIResponse:
    request.state.session_id = session_id
    return await _handle_adapter_request(request, adapter)


@router.post(
    "/v1/messages/count_tokens",
    response_model=anthropic_models.CountMessageTokensResponse,
)
@with_format_chain(
    [FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_CHAT],
    endpoint=UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
async def gemini_messages_count_tokens(
    request: Request,
    body: anthropic_models.CountMessageTokensRequest,
    auth: ConditionalAuthDep,
    adapter: GeminiAdapterDep,
) -> dict[str, int]:
    return await adapter.count_message_tokens(request)


@router.post(
    "/{session_id}/v1/messages/count_tokens",
    response_model=anthropic_models.CountMessageTokensResponse,
)
@with_format_chain(
    [FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_CHAT],
    endpoint=UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
async def gemini_messages_count_tokens_with_session(
    session_id: str,
    request: Request,
    body: anthropic_models.CountMessageTokensRequest,
    auth: ConditionalAuthDep,
    adapter: GeminiAdapterDep,
) -> dict[str, int]:
    request.state.session_id = session_id
    return await adapter.count_message_tokens(request)


@router.get("/v1/models", response_model=openai_models.ModelList)
async def gemini_models(
    request: Request,
    auth: ConditionalAuthDep,
    config: GeminiConfigDep,
) -> dict[str, Any]:
    models = [card.model_dump(mode="json") for card in config.models_endpoint]
    return {"object": "list", "data": models}

