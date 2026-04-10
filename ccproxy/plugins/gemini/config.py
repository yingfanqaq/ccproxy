"""Configuration models for the Gemini provider plugin."""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from ccproxy.models.provider import ModelCard, ModelMappingRule, ProviderConfig

from .model_defaults import (
    DEFAULT_GEMINI_ANTHROPIC_MODEL_TARGETS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_MODEL_CARDS,
    DEFAULT_GEMINI_MODEL_MAPPINGS,
    build_gemini_model_cards,
    build_gemini_model_mappings,
)


DEFAULT_GEMINI_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
GEMINI_OAUTH_CLIENT_ID_ENV = "GEMINI_OAUTH_CLIENT_ID"
GEMINI_OAUTH_CLIENT_SECRET_ENV = "GEMINI_OAUTH_CLIENT_SECRET"
_GEMINI_CLIENT_ID_PATTERNS = (
    re.compile(r'client_id=([A-Za-z0-9._-]+\.apps\.googleusercontent\.com)'),
    re.compile(r'"client_id"\s*:\s*"([A-Za-z0-9._-]+\.apps\.googleusercontent\.com)"'),
)
_GEMINI_CLIENT_SECRET_PATTERNS = (
    re.compile(r'client_secret=([A-Za-z0-9._-]+)'),
    re.compile(r'"client_secret"\s*:\s*"([A-Za-z0-9._-]+)"'),
)
DEFAULT_GEMINI_OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
]


def _read_first_regex_match(path: Path, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    for pattern in patterns:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def _iter_local_gemini_candidate_files() -> list[Path]:
    root = Path.home() / ".gemini"
    candidates: list[Path] = []
    direct_names = [
        "settings.json",
        "oauth_creds.json",
        "google_accounts.json",
    ]

    for name in direct_names:
        path = root / name
        if path.exists():
            candidates.append(path)

    recursive_roots = [root / "tmp", root / "history"]
    for base in recursive_roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                candidates.append(path)

    return candidates


def discover_local_gemini_oauth_client_id() -> str | None:
    for path in _iter_local_gemini_candidate_files():
        value = _read_first_regex_match(path, _GEMINI_CLIENT_ID_PATTERNS)
        if value:
            return value
    return None


def discover_local_gemini_oauth_client_secret() -> str | None:
    for path in _iter_local_gemini_candidate_files():
        value = _read_first_regex_match(path, _GEMINI_CLIENT_SECRET_PATTERNS)
        if value:
            return value
    return None


class AnthropicModelTargets(BaseModel):
    opus: str = Field(default=DEFAULT_GEMINI_ANTHROPIC_MODEL_TARGETS["opus"])
    sonnet: str = Field(default=DEFAULT_GEMINI_ANTHROPIC_MODEL_TARGETS["sonnet"])
    haiku: str = Field(default=DEFAULT_GEMINI_ANTHROPIC_MODEL_TARGETS["haiku"])


class AnthropicEffortMap(BaseModel):
    low: str | None = Field(default=None)
    medium: str | None = Field(default=None)
    high: str | None = Field(default=None)
    max: str | None = Field(default=None)
    adaptive: str | None = Field(default=None)
    disabled: str | None = Field(default=None)


class AnthropicRoutingSettings(BaseModel):
    model_targets: AnthropicModelTargets = Field(
        default_factory=AnthropicModelTargets,
    )
    effort_map: AnthropicEffortMap = Field(
        default_factory=AnthropicEffortMap,
    )


class GeminiConfig(ProviderConfig):
    """Gemini provider configuration."""

    name: str = "gemini"
    base_url: str = "https://cloudcode-pa.googleapis.com/v1internal"
    supports_streaming: bool = True
    requires_auth: bool = True
    auth_type: str | None = "oauth"

    enabled: bool = True
    priority: int = 5
    default_max_tokens: int = 8192

    oauth_client_id: str | None = Field(
        default=None,
        description="Optional OAuth client ID override. Falls back to local Gemini login artifacts or GEMINI_OAUTH_CLIENT_ID.",
    )
    oauth_client_secret: str | None = Field(
        default=None,
        description="Optional OAuth client secret override. Falls back to GEMINI_OAUTH_CLIENT_SECRET or local Gemini artifacts when available.",
    )
    oauth_token_url: str = Field(
        default=DEFAULT_GEMINI_OAUTH_TOKEN_URL,
        description="OAuth token endpoint used to refresh Gemini CLI access tokens.",
    )
    oauth_scopes: list[str] = Field(
        default_factory=lambda: list(DEFAULT_GEMINI_OAUTH_SCOPES),
        description="Scopes reused when refreshing Gemini CLI OAuth credentials.",
    )
    oauth_credentials_path: str = Field(
        default_factory=lambda: str(Path.home() / ".gemini" / "oauth_creds.json"),
        description="Path to Gemini CLI OAuth credentials file.",
    )
    oauth_accounts_path: str = Field(
        default_factory=lambda: str(Path.home() / ".gemini" / "google_accounts.json"),
        description="Path to Gemini CLI account selection file.",
    )
    api_headers: dict[str, str] = Field(
        default_factory=lambda: {
            "Content-Type": "application/json",
        },
        description="Default headers for Gemini Code Assist requests.",
    )
    default_model: str = Field(
        default=DEFAULT_GEMINI_MODEL,
        description="Default upstream Gemini model used by generated model mappings.",
    )
    anthropic_routing: AnthropicRoutingSettings = Field(
        default_factory=AnthropicRoutingSettings,
        description="Configuration table for Anthropic alias routing and effort mapping.",
    )
    model_mappings: list[ModelMappingRule] = Field(
        default_factory=lambda: [
            rule.model_copy(deep=True) for rule in DEFAULT_GEMINI_MODEL_MAPPINGS
        ],
        description="Ordered model translation rules mapping client identifiers to Gemini upstream models.",
    )
    models_endpoint: list[ModelCard] = Field(
        default_factory=lambda: [
            card.model_copy(deep=True) for card in DEFAULT_GEMINI_MODEL_CARDS
        ],
        description="Fallback metadata served from /models.",
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("Gemini base URL must start with http:// or https://")
        return value.rstrip("/")

    @model_validator(mode="after")
    def apply_default_model_overrides(self) -> "GeminiConfig":
        anthropic_targets = self.anthropic_routing.model_targets.model_dump()
        if "model_mappings" not in self.model_fields_set:
            self.model_mappings = build_gemini_model_mappings(
                self.default_model,
                anthropic_model_targets=anthropic_targets,
            )
        if "models_endpoint" not in self.model_fields_set:
            self.models_endpoint = build_gemini_model_cards(
                self.default_model,
                additional_models=anthropic_targets.values(),
            )
        return self

    def resolve_oauth_client_id(self) -> str:
        explicit = self.oauth_client_id or os.getenv(GEMINI_OAUTH_CLIENT_ID_ENV)
        if explicit:
            return explicit

        discovered = discover_local_gemini_oauth_client_id()
        if discovered:
            return discovered

        raise ValueError(
            "Gemini OAuth client_id is required. Set oauth_client_id, "
            "export GEMINI_OAUTH_CLIENT_ID, or log in with the local Gemini CLI first."
        )

    def resolve_oauth_client_secret(self) -> str | None:
        explicit = self.oauth_client_secret or os.getenv(GEMINI_OAUTH_CLIENT_SECRET_ENV)
        if explicit:
            return explicit
        return discover_local_gemini_oauth_client_secret()

    def resolve_credentials_path(self) -> Path:
        return Path(self.oauth_credentials_path).expanduser()

    def resolve_accounts_path(self) -> Path:
        return Path(self.oauth_accounts_path).expanduser()

