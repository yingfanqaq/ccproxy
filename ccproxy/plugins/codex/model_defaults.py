"""Default model metadata and mapping rules for the Codex provider."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ccproxy.models.provider import ModelCard, ModelMappingRule


DEFAULT_CODEX_MODEL = "gpt-5.4"
DEFAULT_CODEX_FAST_MODEL = "gpt-5.4-mini"

DEFAULT_CODEX_ANTHROPIC_MODEL_TARGETS: dict[str, str] = {
    "opus": DEFAULT_CODEX_MODEL,
    "sonnet": DEFAULT_CODEX_MODEL,
    "haiku": DEFAULT_CODEX_FAST_MODEL,
}

_CODEX_MODEL_METADATA: dict[str, dict[str, object]] = {
    "gpt-5.4": {
        "created": 1743465600,
        "owned_by": "openai",
    },
    "gpt-5.4-mini": {
        "created": 1743465600,
        "owned_by": "openai",
    },
    "gpt-5.3-codex": {
        "created": 1723075200,
        "owned_by": "openai",
    },
    "gpt-5.2-codex": {
        "created": 1726444800,
        "owned_by": "openai",
    },
}


def _ordered_model_ids(
    default_model: str,
    additional_models: Iterable[str] | None = None,
) -> list[str]:
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for model_id in [default_model, *(additional_models or []), *_CODEX_MODEL_METADATA]:
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            continue
        seen.add(model_id)
        ordered_ids.append(model_id)
    return ordered_ids


def build_codex_model_cards(
    default_model: str = DEFAULT_CODEX_MODEL,
    additional_models: Iterable[str] | None = None,
) -> list[ModelCard]:
    cards: list[ModelCard] = []
    for model_id in _ordered_model_ids(default_model, additional_models):
        metadata = _CODEX_MODEL_METADATA.get(model_id)
        if metadata is None:
            metadata = {"created": None, "owned_by": "openai"}
        cards.append(
            ModelCard(
                id=model_id,
                created=metadata["created"],
                owned_by=metadata["owned_by"],
                permission=[],
                root=model_id,
                parent=None,
            )
        )
    return cards


def build_codex_model_mappings(
    default_model: str = DEFAULT_CODEX_MODEL,
    anthropic_model_targets: Mapping[str, str] | None = None,
) -> list[ModelMappingRule]:
    resolved_targets = dict(DEFAULT_CODEX_ANTHROPIC_MODEL_TARGETS)
    if anthropic_model_targets is not None:
        for family in ("opus", "sonnet", "haiku"):
            target = anthropic_model_targets.get(family)
            if isinstance(target, str) and target:
                resolved_targets[family] = target

    return [
        ModelMappingRule(match="gpt-5", target=default_model),
        ModelMappingRule(match="gpt-5-codex", target=default_model),
        ModelMappingRule(
            match="^claude-.*opus",
            target=resolved_targets["opus"],
            kind="regex",
            flags=["IGNORECASE"],
        ),
        ModelMappingRule(
            match="^claude-.*sonnet",
            target=resolved_targets["sonnet"],
            kind="regex",
            flags=["IGNORECASE"],
        ),
        ModelMappingRule(
            match="^claude-.*haiku",
            target=resolved_targets["haiku"],
            kind="regex",
            flags=["IGNORECASE"],
        ),
        ModelMappingRule(match="opus", target=resolved_targets["opus"]),
        ModelMappingRule(match="sonnet", target=resolved_targets["sonnet"]),
        ModelMappingRule(match="haiku", target=resolved_targets["haiku"]),
    ]


DEFAULT_CODEX_MODEL_CARDS: list[ModelCard] = build_codex_model_cards(
    additional_models=DEFAULT_CODEX_ANTHROPIC_MODEL_TARGETS.values()
)


DEFAULT_CODEX_MODEL_MAPPINGS: list[ModelMappingRule] = build_codex_model_mappings()


__all__ = [
    "DEFAULT_CODEX_ANTHROPIC_MODEL_TARGETS",
    "DEFAULT_CODEX_FAST_MODEL",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_MODEL_CARDS",
    "DEFAULT_CODEX_MODEL_MAPPINGS",
    "build_codex_model_cards",
    "build_codex_model_mappings",
]
