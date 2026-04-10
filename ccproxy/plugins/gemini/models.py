"""Credential models for the Gemini provider plugin."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_serializer, field_validator


class GeminiOAuthCredentials(BaseModel):
    """Google OAuth credentials stored by the Gemini CLI."""

    access_token: SecretStr = Field(..., description="OAuth access token")
    scope: str | None = Field(default=None, description="Granted OAuth scopes")
    token_type: str = Field(default="Bearer", description="OAuth token type")
    id_token: SecretStr | None = Field(default=None, description="Optional ID token")
    expiry_date: int | None = Field(
        default=None,
        description="Token expiration in Unix milliseconds",
    )
    refresh_token: SecretStr | None = Field(
        default=None,
        description="OAuth refresh token",
    )
    account_email: str | None = Field(
        default=None,
        description="Active Google account email inferred from Gemini CLI state",
    )

    @field_serializer("access_token", "id_token", "refresh_token")
    def _serialize_secret(self, value: SecretStr | None) -> str | None:
        return value.get_secret_value() if value is not None else None

    @field_validator("access_token", "id_token", "refresh_token", mode="before")
    @classmethod
    def _coerce_secret(
        cls, value: str | SecretStr | None
    ) -> SecretStr | None:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            return value
        return SecretStr(str(value))

    @property
    def access_token_value(self) -> str:
        return self.access_token.get_secret_value()

    @property
    def refresh_token_value(self) -> str | None:
        if self.refresh_token is None:
            return None
        return self.refresh_token.get_secret_value()

    @property
    def id_token_value(self) -> str | None:
        if self.id_token is None:
            return None
        return self.id_token.get_secret_value()

    @property
    def expires_at(self) -> datetime | None:
        if self.expiry_date is None:
            return None
        return datetime.fromtimestamp(self.expiry_date / 1000, tz=UTC)

    def is_expired(self) -> bool:
        expires_at = self.expires_at
        if expires_at is None:
            return False
        return datetime.now(UTC) >= expires_at

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeminiOAuthCredentials":
        return cls(**data)
