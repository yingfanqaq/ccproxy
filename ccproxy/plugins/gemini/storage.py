"""Token storage for Gemini CLI OAuth credentials."""

from __future__ import annotations

from pathlib import Path

from ccproxy.auth.storage.base import BaseJsonStorage
from ccproxy.core.logging import get_plugin_logger

from .models import GeminiOAuthCredentials


logger = get_plugin_logger()


class GeminiTokenStorage(BaseJsonStorage[GeminiOAuthCredentials]):
    """Storage wrapper for ``~/.gemini/oauth_creds.json``."""

    def __init__(self, storage_path: Path | None = None):
        if storage_path is None:
            storage_path = Path.home() / ".gemini" / "oauth_creds.json"
        super().__init__(storage_path)
        self.provider_name = "gemini"

    async def save(self, credentials: GeminiOAuthCredentials) -> bool:
        try:
            await self._write_json(credentials.model_dump(mode="json", exclude_none=True))
            logger.info(
                "gemini_oauth_credentials_saved",
                storage_path=str(self.file_path),
                has_refresh_token=bool(credentials.refresh_token_value),
                category="auth",
            )
            return True
        except Exception as exc:
            logger.error(
                "gemini_oauth_save_failed",
                error=str(exc),
                exc_info=exc,
                category="auth",
            )
            return False

    async def load(self) -> GeminiOAuthCredentials | None:
        try:
            data = await self._read_json()
            if not data:
                return None
            credentials = GeminiOAuthCredentials.model_validate(data)
            logger.debug(
                "gemini_oauth_credentials_loaded",
                has_refresh_token=bool(credentials.refresh_token_value),
                category="auth",
            )
            return credentials
        except Exception as exc:
            logger.error(
                "gemini_oauth_credentials_load_failed",
                error=str(exc),
                exc_info=exc,
                category="auth",
            )
            return None
