"""Gemini token manager backed by Gemini CLI login state."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import time
from typing import Any

import httpx

from ccproxy.auth.managers.base import BaseTokenManager
from ccproxy.auth.managers.token_snapshot import TokenSnapshot
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_plugin_logger

from .models import GeminiOAuthCredentials
from .storage import GeminiTokenStorage


logger = get_plugin_logger()


class GeminiTokenManager(BaseTokenManager[GeminiOAuthCredentials]):
    """Manager for OAuth credentials created by ``gemini login``."""

    def __init__(
        self,
        storage: TokenStorage[GeminiOAuthCredentials] | None = None,
        *,
        client_id: str,
        client_secret: str | None,
        token_url: str,
        scopes: list[str] | None = None,
        accounts_path: Path | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        storage = storage or GeminiTokenStorage()
        super().__init__(storage)
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scopes = list(scopes or [])
        self.accounts_path = accounts_path or (Path.home() / ".gemini" / "google_accounts.json")
        self._http_client = http_client
        self._owns_http_client = http_client is None

    @classmethod
    async def create(
        cls,
        storage: TokenStorage[GeminiOAuthCredentials] | None = None,
        *,
        client_id: str,
        client_secret: str | None,
        token_url: str,
        scopes: list[str] | None = None,
        accounts_path: Path | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> "GeminiTokenManager":
        return cls(
            storage=storage,
            client_id=client_id,
            client_secret=client_secret,
            token_url=token_url,
            scopes=scopes,
            accounts_path=accounts_path,
            http_client=http_client,
        )

    def _build_token_snapshot(
        self, credentials: GeminiOAuthCredentials
    ) -> TokenSnapshot:
        extras = {
            "token_type": credentials.token_type,
            "scope": credentials.scope,
            "account_email": credentials.account_email,
            "id_token_present": bool(credentials.id_token_value),
        }
        return TokenSnapshot(
            provider="gemini",
            account_id=credentials.account_email,
            access_token=credentials.access_token_value,
            refresh_token=credentials.refresh_token_value,
            expires_at=credentials.expires_at,
            extras=extras,
        )

    async def refresh_token(self) -> GeminiOAuthCredentials | None:
        credentials = await self.load_credentials()
        if not credentials:
            logger.error("gemini_refresh_no_credentials", category="auth")
            return None

        refresh_token = credentials.refresh_token_value
        if not refresh_token:
            logger.error("gemini_refresh_no_refresh_token", category="auth")
            return None

        form_data: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }
        if self.client_secret:
            form_data["client_secret"] = self.client_secret
        if self.scopes:
            form_data["scope"] = " ".join(self.scopes)

        client = await self._get_http_client()
        try:
            response = await client.post(
                self.token_url,
                data=form_data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            response_text = ""
            try:
                response_text = exc.response.text[:500]
            except Exception:
                response_text = ""
            logger.error(
                "gemini_token_refresh_failed",
                error=str(exc),
                response_text=response_text,
                exc_info=exc,
                category="auth",
            )
            return None
        except Exception as exc:
            logger.error(
                "gemini_token_refresh_failed",
                error=str(exc),
                exc_info=exc,
                category="auth",
            )
            return None

        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            logger.error(
                "gemini_token_refresh_missing_access_token",
                category="auth",
            )
            return None

        expires_in = payload.get("expires_in")
        expiry_date: int | None
        if isinstance(expires_in, (int, float)):
            expiry_date = int((time() + float(expires_in)) * 1000)
        else:
            expiry_date = credentials.expiry_date

        refreshed = GeminiOAuthCredentials(
            access_token=access_token,
            refresh_token=payload.get("refresh_token") or refresh_token,
            id_token=payload.get("id_token") or credentials.id_token_value,
            token_type=str(payload.get("token_type") or credentials.token_type or "Bearer"),
            scope=str(payload.get("scope") or credentials.scope or ""),
            expiry_date=expiry_date,
            account_email=credentials.account_email or await self._load_active_account(),
        )

        if await self.save_credentials(refreshed):
            self._profile_cache = None
            return refreshed
        return None

    def is_expired(self, credentials: GeminiOAuthCredentials) -> bool:
        return credentials.is_expired()

    def get_account_id(self, credentials: GeminiOAuthCredentials) -> str | None:
        return credentials.account_email

    def get_expiration_time(
        self, credentials: GeminiOAuthCredentials
    ) -> datetime | None:
        return credentials.expires_at

    async def get_access_token(self) -> str | None:
        credentials = await self.load_credentials()
        if not credentials:
            return None

        if not credentials.account_email:
            credentials.account_email = await self._load_active_account()
            self._credentials_cache = credentials

        if self.should_refresh(credentials):
            refreshed = await self.get_access_token_with_refresh()
            if refreshed:
                return refreshed

        return credentials.access_token_value

    async def get_access_token_with_refresh(self) -> str | None:
        credentials = await self.load_credentials()
        if not credentials:
            return None

        if not self.should_refresh(credentials):
            return credentials.access_token_value

        refreshed = await self.refresh_token()
        if refreshed:
            return refreshed.access_token_value
        return credentials.access_token_value

    async def aclose(self) -> None:
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _load_active_account(self) -> str | None:
        path = self.accounts_path
        if not path.exists():
            return None

        try:
            def _read_json() -> dict[str, Any]:
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)

            payload = await asyncio.to_thread(_read_json)
        except Exception as exc:
            logger.debug(
                "gemini_accounts_load_failed",
                path=str(path),
                error=str(exc),
                category="auth",
            )
            return None

        active = payload.get("active")
        return str(active) if isinstance(active, str) and active else None
