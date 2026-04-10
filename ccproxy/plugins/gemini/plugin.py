"""Gemini provider plugin implementation."""

from __future__ import annotations

from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
)
from ccproxy.core.plugins import (
    BaseProviderPluginFactory,
    FormatPair,
    PluginContext,
    PluginManifest,
    ProviderPluginRuntime,
)
from ccproxy.core.plugins.declaration import RouterSpec
from ccproxy.llms.streaming.accumulators import OpenAIAccumulator

from .adapter import GeminiAdapter
from .config import GeminiConfig
from .manager import GeminiTokenManager
from .routes import router as gemini_router
from .storage import GeminiTokenStorage


class GeminiRuntime(ProviderPluginRuntime):
    """Runtime for the Gemini provider plugin."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.config: GeminiConfig | None = None

    async def _on_initialize(self) -> None:
        if not self.context:
            raise RuntimeError("Context not set")

        config = self.context.get("config")
        if not isinstance(config, GeminiConfig):
            config = GeminiConfig()
        self.config = config

        await super()._on_initialize()


class GeminiFactory(BaseProviderPluginFactory):
    """Factory for the Gemini provider plugin."""

    cli_safe = False

    plugin_name = "gemini"
    plugin_description = "Google Gemini provider plugin using Gemini CLI login and Code Assist upstream"
    runtime_class = GeminiRuntime
    adapter_class = GeminiAdapter
    config_class = GeminiConfig
    routers = [
        RouterSpec(router=gemini_router, prefix="/gemini"),
    ]
    dependencies = []
    optional_requires = []
    requires_format_adapters: list[FormatPair] = [
        (FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_CHAT),
        (FORMAT_OPENAI_CHAT, FORMAT_ANTHROPIC_MESSAGES),
        (FORMAT_OPENAI_RESPONSES, FORMAT_OPENAI_CHAT),
        (FORMAT_OPENAI_CHAT, FORMAT_OPENAI_RESPONSES),
    ]
    tool_accumulator_class = OpenAIAccumulator

    async def create_credentials_manager(self, context: PluginContext) -> GeminiTokenManager:
        config = context.get("config")
        if not isinstance(config, GeminiConfig):
            config = GeminiConfig()

        return await GeminiTokenManager.create(
            storage=GeminiTokenStorage(config.resolve_credentials_path()),
            client_id=config.resolve_oauth_client_id(),
            client_secret=config.resolve_oauth_client_secret(),
            token_url=config.oauth_token_url,
            scopes=config.oauth_scopes,
            accounts_path=config.resolve_accounts_path(),
        )


factory = GeminiFactory()
