"""HTTP adapter for Google Gemini Code Assist."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Mapping
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_RESPONSES,
)
from ccproxy.core.errors import AuthenticationError
from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.streaming import DeferredStreaming
from ccproxy.streaming.sse import serialize_json_to_sse_stream
from ccproxy.utils.headers import (
    extract_request_headers,
    extract_response_headers,
    filter_request_headers,
    filter_response_headers,
)
from ccproxy.utils.model_mapper import restore_model_aliases

from .code_assist import (
    build_client_metadata,
    code_assist_stream_to_openai_chat_chunks,
    code_assist_to_openai_chat_response,
    collect_tool_signatures_from_code_assist_payload,
    openai_chat_chunks_to_anthropic_events,
    openai_chat_chunks_to_responses_events,
    openai_chat_to_code_assist_count_request,
    openai_chat_to_code_assist_request,
)
from .config import GeminiConfig


logger = get_plugin_logger()


class GeminiAdapter(BaseHTTPAdapter):
    """HTTP adapter for Gemini's OpenAI-compatible API."""

    def __init__(
        self,
        config: GeminiConfig,
        auth_manager: Any,
        http_pool_manager: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config=config,
            auth_manager=auth_manager,
            http_pool_manager=http_pool_manager,
            **kwargs,
        )
        self.base_url = self.config.base_url.rstrip("/")
        self.token_manager = auth_manager
        self._code_assist_context: dict[str, Any] | None = None
        self._code_assist_lock = asyncio.Lock()
        self._tool_signature_cache: dict[str, str] = {}

    def _base_origin(self) -> str:
        parsed = urlparse(self.base_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _normalize_error_payload(
        self,
        payload: Any,
        *,
        status_code: int,
        default_message: str,
    ) -> dict[str, Any]:
        message = default_message
        details = payload
        if isinstance(payload, Mapping):
            error_payload = payload.get("error")
            if isinstance(error_payload, Mapping):
                message = str(
                    error_payload.get("message")
                    or error_payload.get("status")
                    or default_message
                )
            else:
                message = str(payload.get("message") or default_message)
        elif payload is not None:
            message = str(payload)

        error_type = "authentication_error" if status_code in {401, 403} else "api_error"
        return {
            "error": {
                "message": message,
                "type": error_type,
                "details": details,
            }
        }

    def _code_assist_error_response(
        self,
        payload: Any,
        *,
        status_code: int,
        default_message: str,
    ) -> Response:
        normalized = self._normalize_error_payload(
            payload,
            status_code=status_code,
            default_message=default_message,
        )
        return Response(
            content=json.dumps(normalized).encode(),
            status_code=status_code,
            media_type="application/json",
        )

    async def _load_code_assist_context(self, *, force: bool = False) -> dict[str, Any]:
        async with self._code_assist_lock:
            if self._code_assist_context is not None and not force:
                return dict(self._code_assist_context)

            access_token = await self._resolve_access_token()
            client = await self.http_pool_manager.get_client(base_url=self._base_origin())
            response = await client.post(
                f"{self.base_url}:loadCodeAssist",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {access_token}",
                },
                content=json.dumps({"metadata": build_client_metadata()}).encode(),
            )

            try:
                payload = response.json()
            except Exception:
                payload = response.text

            if response.status_code >= 400:
                normalized = self._normalize_error_payload(
                    payload,
                    status_code=response.status_code,
                    default_message="Failed to load Gemini Code Assist context",
                )
                raise AuthenticationError(
                    normalized["error"]["message"]
                )

            if not isinstance(payload, dict):
                raise AuthenticationError("Invalid Gemini Code Assist context payload")

            self._code_assist_context = payload
            return dict(payload)

    async def _get_project_id(self) -> str | None:
        context = await self._load_code_assist_context()
        project_id = context.get("cloudaicompanionProject")
        return str(project_id) if isinstance(project_id, str) and project_id else None

    def _remember_tool_signatures(self, payload: Mapping[str, Any]) -> None:
        for tool_call_id, signature in collect_tool_signatures_from_code_assist_payload(payload).items():
            self._tool_signature_cache[tool_call_id] = signature

    def _log_missing_thought_signatures(
        self, payload: Mapping[str, Any]
    ) -> None:
        request_payload = payload.get("request")
        if not isinstance(request_payload, Mapping):
            return
        contents = request_payload.get("contents")
        if not isinstance(contents, list):
            return

        missing: list[dict[str, Any]] = []
        for content_index, content in enumerate(contents):
            if not isinstance(content, Mapping):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part_index, part in enumerate(parts):
                if not isinstance(part, Mapping):
                    continue
                function_call = part.get("functionCall")
                if not isinstance(function_call, Mapping):
                    continue
                if function_call.get("thoughtSignature"):
                    continue
                missing.append(
                    {
                        "content_index": content_index,
                        "part_index": part_index,
                        "id": function_call.get("id"),
                        "name": function_call.get("name"),
                    }
                )

        if missing:
            logger.warning(
                "gemini_request_missing_thought_signature",
                total=len(missing),
                missing=missing[:10],
                category="transform",
            )

    async def _streaming_json_payloads(
        self,
        response: httpx.Response,
    ):
        buffer: list[str] = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                buffer.append(line[6:].strip())
                continue
            if line == "":
                if not buffer:
                    continue
                chunk = "\n".join(buffer)
                buffer = []
                try:
                    payload = json.loads(chunk)
                except Exception:
                    continue
                if isinstance(payload, Mapping):
                    self._remember_tool_signatures(payload)
                    yield payload
        if buffer:
            try:
                payload = json.loads("\n".join(buffer))
            except Exception:
                payload = None
            if isinstance(payload, Mapping):
                self._remember_tool_signatures(payload)
                yield payload

    async def _single_payload_stream(self, payload: dict[str, Any]):
        yield payload


    async def handle_request(
        self, request: Request
    ) -> Response | StreamingResponse | DeferredStreaming:
        ctx = request.state.context
        self._ensure_tool_accumulator(ctx)

        body = await request.body()
        body = await self._map_request_model(ctx, body)
        headers = extract_request_headers(request)
        method = request.method
        endpoint = ctx.metadata.get("endpoint", "")

        self._ensure_format_registry(ctx.format_chain, endpoint)

        if self.streaming_handler:
            body_wants_stream = False
            try:
                parsed_payload = json.loads(body.decode()) if body else {}
                body_wants_stream = bool(parsed_payload.get("stream", False))
            except Exception:
                body_wants_stream = False
            header_wants_stream = self.streaming_handler.should_stream_response(headers)
            if body_wants_stream or header_wants_stream:
                try:
                    parsed_payload = json.loads(body.decode()) if body else {}
                    if isinstance(parsed_payload, dict):
                        self._record_tool_definitions(ctx, parsed_payload)
                except Exception:
                    pass
                return await self.handle_streaming(request, endpoint)

        if ctx.format_chain and len(ctx.format_chain) > 1:
            try:
                source_payload = self._decode_json_body(body, context="request")
                request_payload = source_payload
                request_payload = await self._apply_format_chain(
                    data=request_payload,
                    format_chain=ctx.format_chain,
                    stage="request",
                )
                request_payload = self._apply_anthropic_routing(
                    ctx, source_payload, request_payload
                )
                self._record_tool_definitions(ctx, request_payload)
                body = self._encode_json_body(request_payload)
                logger.info(
                    "format_chain_applied",
                    stage="request",
                    endpoint=endpoint,
                    chain=ctx.format_chain,
                    steps=len(ctx.format_chain) - 1,
                    category="format",
                )
            except Exception as exc:
                logger.error(
                    "format_chain_request_failed",
                    error=str(exc),
                    endpoint=endpoint,
                    exc_info=exc,
                    category="transform",
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Failed to convert request using format chain",
                            "details": str(exc),
                        }
                    },
                )

        prepared_body, prepared_headers = await self.prepare_provider_request(
            body,
            headers,
            endpoint,
            session_id=getattr(request.state, "session_id", None),
        )
        target_url = await self.get_target_url(endpoint, streaming=False)
        (
            method,
            target_url,
            prepared_body,
            prepared_headers,
        ) = await self._emit_provider_request_prepared(
            request_obj=request,
            ctx=ctx,
            method=method,
            endpoint=endpoint,
            target_url=target_url,
            prepared_body=prepared_body,
            prepared_headers=prepared_headers,
            is_streaming=False,
        )
        provider_response = await self._execute_http_request(
            method,
            target_url,
            prepared_headers,
            prepared_body,
        )

        response = await self.process_provider_response(provider_response, endpoint)
        headers = filter_response_headers(dict(provider_response.headers))

        if isinstance(response, StreamingResponse):
            return await self._convert_streaming_response(response, ctx.format_chain, ctx)
        if isinstance(response, Response):
            response = self._restore_model_response(response, ctx)
            for header in ("content-encoding", "transfer-encoding", "content-length"):
                with contextlib.suppress(KeyError):
                    del response.headers[header]
            if ctx.format_chain and len(ctx.format_chain) > 1:
                stage: Literal["response", "error"] = (
                    "error" if provider_response.status_code >= 400 else "response"
                )
                try:
                    payload = self._decode_json_body(cast(bytes, response.body), context=stage)
                    payload = await self._apply_format_chain(
                        data=payload,
                        format_chain=ctx.format_chain,
                        stage=stage,
                    )
                    metadata = getattr(ctx, "metadata", None)
                    if isinstance(metadata, dict):
                        alias_map = metadata.get("_model_alias_map")
                    else:
                        alias_map = None
                    if not alias_map:
                        alias_map = getattr(ctx, "_model_alias_map", None)
                    if isinstance(metadata, dict):
                        if (
                            isinstance(payload, dict)
                            and isinstance(alias_map, Mapping)
                            and isinstance(payload.get("model"), str)
                        ):
                            payload["model"] = alias_map.get(
                                payload["model"], payload["model"]
                            )
                        restore_model_aliases(payload, metadata)
                    body_bytes = self._encode_json_body(payload)
                    restored = Response(
                        content=body_bytes,
                        status_code=provider_response.status_code,
                        headers=headers,
                        media_type=provider_response.headers.get(
                            "content-type", "application/json"
                        ),
                    )
                    return self._restore_model_response(restored, ctx)
                except Exception as exc:
                    logger.error(
                        "format_chain_response_failed",
                        error=str(exc),
                        endpoint=endpoint,
                        stage=stage,
                        exc_info=exc,
                        category="transform",
                    )
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": {
                                "type": "internal_server_error",
                                "message": "Failed to convert response format",
                                "details": str(exc),
                            }
                        },
                    )
            return self._restore_model_response(response, ctx)

        restored = Response(
            content=provider_response.content,
            status_code=provider_response.status_code,
            headers=headers,
            media_type=headers.get("content-type", "application/json"),
        )
        return self._restore_model_response(restored, ctx)

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse | DeferredStreaming:
        ctx = request.state.context
        body = await request.body()
        body = await self._map_request_model(ctx, body)
        headers = extract_request_headers(request)
        method = request.method

        self._ensure_tool_accumulator(ctx)
        self._ensure_format_registry(ctx.format_chain, endpoint)

        if ctx.format_chain and len(ctx.format_chain) > 1:
            try:
                source_payload = self._decode_json_body(body, context="stream_request")
                stream_payload = await self._apply_format_chain(
                    data=source_payload,
                    format_chain=ctx.format_chain,
                    stage="request",
                )
                stream_payload = self._apply_anthropic_routing(
                    ctx, source_payload, stream_payload
                )
                self._record_tool_definitions(ctx, stream_payload)
                body = self._encode_json_body(stream_payload)
            except Exception as exc:
                logger.error(
                    "format_chain_stream_request_failed",
                    error=str(exc),
                    endpoint=endpoint,
                    exc_info=exc,
                    category="transform",
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Failed to convert streaming request using format chain",
                            "details": str(exc),
                        }
                    },
                )

        prepared_body, prepared_headers = await self.prepare_provider_request(
            body,
            headers,
            endpoint,
            session_id=getattr(request.state, "session_id", None),
        )
        target_url = await self.get_target_url(endpoint, streaming=True)
        (
            method,
            target_url,
            prepared_body,
            prepared_headers,
        ) = await self._emit_provider_request_prepared(
            request_obj=request,
            ctx=ctx,
            method=method,
            endpoint=endpoint,
            target_url=target_url,
            prepared_body=prepared_body,
            prepared_headers=prepared_headers,
            is_streaming=True,
        )

        async def final_stream():
            client = await self.http_pool_manager.get_streaming_client(
                base_url=self._base_origin()
            )
            async with client.stream(
                method,
                target_url,
                headers=prepared_headers,
                content=prepared_body,
            ) as provider_response:
                if provider_response.status_code >= 400:
                    try:
                        error_payload = await provider_response.aread()
                        parsed_error = json.loads(error_payload.decode()) if error_payload else {}
                    except Exception:
                        parsed_error = {"error": {"message": "Gemini streaming request failed"}}

                    normalized = self._normalize_error_payload(
                        parsed_error,
                        status_code=provider_response.status_code,
                        default_message="Gemini Code Assist streaming request failed",
                    )
                    target_format = ctx.format_chain[0] if ctx.format_chain else None
                    if target_format == FORMAT_ANTHROPIC_MESSAGES:
                        error_event = {
                            "type": "error",
                            "error": normalized["error"],
                        }
                    else:
                        error_event = normalized
                    async for chunk in serialize_json_to_sse_stream(
                        self._single_payload_stream(error_event),
                        include_done=False,
                        request_context=ctx,
                    ):
                        yield chunk
                    return

                openai_stream = code_assist_stream_to_openai_chat_chunks(
                    self._streaming_json_payloads(provider_response)
                )
                target_format = ctx.format_chain[0] if ctx.format_chain else None
                if target_format == FORMAT_ANTHROPIC_MESSAGES:
                    payload_stream = openai_chat_chunks_to_anthropic_events(openai_stream)
                elif target_format == FORMAT_OPENAI_RESPONSES:
                    payload_stream = openai_chat_chunks_to_responses_events(openai_stream)
                else:
                    async def openai_payload_stream():
                        async for chunk in openai_stream:
                            yield chunk.model_dump(mode="json", exclude_none=True)
                    payload_stream = openai_payload_stream()

                async for chunk in serialize_json_to_sse_stream(
                    payload_stream,
                    include_done=True,
                    request_context=ctx,
                ):
                    yield chunk

        return StreamingResponse(
            content=final_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    async def get_target_url(
        self,
        endpoint: str,
        *,
        streaming: bool = False,
        count_tokens: bool = False,
    ) -> str:
        if count_tokens:
            return f"{self.base_url}:countTokens"
        if streaming:
            return f"{self.base_url}:streamGenerateContent?alt=sse"
        return f"{self.base_url}:generateContent"

    def _apply_anthropic_routing(
        self,
        ctx: Any,
        source_payload: dict[str, Any],
        converted_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return converted_payload

    def _sanitize_provider_body(self, body_data: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(body_data)
        for key in ("metadata", "reasoning_effort", "service_tier"):
            sanitized.pop(key, None)
        if sanitized.get("stream_options") is None:
            sanitized.pop("stream_options", None)
        return sanitized

    async def count_message_tokens(
        self,
        request: Request,
    ) -> dict[str, int]:
        ctx = request.state.context
        raw_body = await request.body()
        raw_body = await self._map_request_model(ctx, raw_body)
        source_payload = self._decode_json_body(raw_body, context="count_request")
        source_payload.setdefault("max_tokens", 1)
        source_payload.setdefault("stream", False)

        request_payload = source_payload
        if ctx.format_chain and len(ctx.format_chain) > 1:
            request_payload = await self._apply_format_chain(
                data=request_payload,
                format_chain=ctx.format_chain,
                stage="request",
            )
            request_payload = self._apply_anthropic_routing(
                ctx, source_payload, request_payload
            )

        if not isinstance(request_payload, Mapping):
            raise HTTPException(status_code=400, detail="Invalid count_tokens payload")

        count_request = openai_chat_to_code_assist_count_request(request_payload)
        access_token = await self._resolve_access_token()
        client = await self.http_pool_manager.get_client(base_url=self._base_origin())
        response = await client.post(
            await self.get_target_url("countTokens", count_tokens=True),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
            content=json.dumps(count_request).encode(),
        )
        try:
            payload = response.json()
        except Exception:
            payload = response.text

        if response.status_code >= 400:
            normalized = self._normalize_error_payload(
                payload,
                status_code=response.status_code,
                default_message="Gemini Code Assist countTokens failed",
            )
            raise HTTPException(status_code=response.status_code, detail=normalized)

        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=502, detail="Invalid countTokens response")

        return {"input_tokens": int(payload.get("totalTokens") or 0)}


    async def _resolve_access_token(self) -> str:
        token_manager = self.token_manager
        if token_manager is None:
            raise AuthenticationError(
                "Authentication manager not configured for Gemini provider"
            )

        async def _snapshot_token() -> str | None:
            snapshot = await token_manager.get_token_snapshot()
            if snapshot and snapshot.access_token:
                return str(snapshot.access_token)
            return None

        credentials = await token_manager.load_credentials()
        if not credentials:
            fallback = await _snapshot_token()
            if fallback:
                return fallback
            raise ValueError("No Gemini credentials available")

        try:
            if token_manager.should_refresh(credentials):
                logger.debug("gemini_token_refresh_due", category="auth")
                refreshed = await token_manager.get_access_token_with_refresh()
                if refreshed:
                    return refreshed
        except Exception as exc:
            logger.warning(
                "gemini_token_refresh_failed",
                error=str(exc),
                category="auth",
            )
            fallback = await _snapshot_token()
            if fallback:
                return fallback

        try:
            token = await token_manager.get_access_token()
            if token:
                return token
        except Exception as exc:
            logger.warning(
                "gemini_token_fetch_failed",
                error=str(exc),
                category="auth",
            )

        fallback = await _snapshot_token()
        if fallback:
            return fallback

        raise ValueError("No valid Gemini access token available")

    async def prepare_provider_request(
        self,
        body: bytes,
        headers: dict[str, str],
        endpoint: str,
        *,
        session_id: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        access_token = await self._resolve_access_token()
        project_id = await self._get_project_id()

        try:
            parsed_body = json.loads(body.decode()) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed_body = None

        if not isinstance(parsed_body, dict):
            raise HTTPException(status_code=400, detail="Invalid Gemini request payload")

        sanitized_payload = self._sanitize_provider_body(parsed_body)
        code_assist_payload = openai_chat_to_code_assist_request(
            sanitized_payload,
            project_id=project_id,
            session_id=session_id,
            thought_signatures=self._tool_signature_cache,
        )
        self._log_missing_thought_signatures(code_assist_payload)
        body = json.dumps(code_assist_payload).encode()

        filtered_headers = filter_request_headers(headers, preserve_auth=False)
        provider_headers = {
            key: str(value)
            for key, value in self.config.api_headers.items()
            if value is not None
        }
        provider_headers["Authorization"] = f"Bearer {access_token}"
        provider_headers.setdefault("Content-Type", "application/json")
        final_headers = {**filtered_headers, **provider_headers}
        return body, final_headers

    async def process_provider_response(
        self, response: httpx.Response, endpoint: str
    ) -> Response:
        response_headers = extract_response_headers(response)

        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            payload = None

        if response.status_code >= 400:
            return self._code_assist_error_response(
                payload if payload is not None else response.text,
                status_code=response.status_code,
                default_message="Gemini Code Assist request failed",
            )

        if not isinstance(payload, Mapping):
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get("content-type", "application/json"),
            )

        self._remember_tool_signatures(payload)
        openai_payload = code_assist_to_openai_chat_response(payload)
        return Response(
            content=json.dumps(openai_payload).encode(),
            status_code=response.status_code,
            headers=response_headers,
            media_type="application/json",
        )
