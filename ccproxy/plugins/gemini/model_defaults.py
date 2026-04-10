"""Default model metadata and mapping rules for the Gemini provider."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ccproxy.models.provider import ModelCard, ModelMappingRule


DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_GEMINI_FAST_MODEL = "gemini-3-flash-preview"
DEFAULT_GEMINI_LIGHT_MODEL = "gemini-3.1-flash-lite-preview"

DEFAULT_GEMINI_ANTHROPIC_MODEL_TARGETS: dict[str, str] = {
    "opus": DEFAULT_GEMINI_MODEL,
    "sonnet": DEFAULT_GEMINI_FAST_MODEL,
    "haiku": DEFAULT_GEMINI_LIGHT_MODEL,
}

_GEMINI_MODEL_METADATA: dict[str, dict[str, object]] = {
    "gemini-3.1-pro-preview": {
        "created": 1775001600,
        "owned_by": "google",
    },
    "gemini-3-flash-preview": {
        "created": 1775001600,
        "owned_by": "google",
    },
    "gemini-3.1-flash-lite-preview": {
        "created": 1775001600,
        "owned_by": "google",
    },
    "gemini-2.5-pro": {
        "created": 1743465600,
        "owned_by": "google",
    },
    "gemini-2.5-flash": {
        "created": 1743465600,
        "owned_by": "google",
    },
    "gemini-2.5-flash-lite": {
        "created": 1743465600,
        "owned_by": "google",
    },
}


def _ordered_model_ids(
    default_model: str,
    additional_models: Iterable[str] | None = None,
) -> list[str]:
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for model_id in [default_model, *(additional_models or []), *_GEMINI_MODEL_METADATA]:
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            continue
        seen.add(model_id)
        ordered_ids.append(model_id)
    return ordered_ids


def build_gemini_model_cards(
    default_model: str = DEFAULT_GEMINI_MODEL,
    additional_models: Iterable[str] | None = None,
) -> list[ModelCard]:
    cards: list[ModelCard] = []
    for model_id in _ordered_model_ids(default_model, additional_models):
        metadata = _GEMINI_MODEL_METADATA.get(model_id)
        if metadata is None:
            metadata = {"created": None, "owned_by": "google"}
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


def build_gemini_model_mappings(
    default_model: str = DEFAULT_GEMINI_MODEL,
    anthropic_model_targets: Mapping[str, str] | None = None,
) -> list[ModelMappingRule]:
    resolved_targets = dict(DEFAULT_GEMINI_ANTHROPIC_MODEL_TARGETS)
    if anthropic_model_targets is not None:
        for family in ("opus", "sonnet", "haiku"):
            target = anthropic_model_targets.get(family)
            if isinstance(target, str) and target:
                resolved_targets[family] = target

    return [
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
        ModelMappingRule(
            match=r"^opus(?:\[[^\]]+\])?$",
            target=resolved_targets["opus"],
            kind="regex",
            flags=["IGNORECASE"],
        ),
        ModelMappingRule(
            match=r"^sonnet(?:\[[^\]]+\])?$",
            target=resolved_targets["sonnet"],
            kind="regex",
            flags=["IGNORECASE"],
        ),
        ModelMappingRule(
            match=r"^haiku(?:\[[^\]]+\])?$",
            target=resolved_targets["haiku"],
            kind="regex",
            flags=["IGNORECASE"],
        ),
        ModelMappingRule(match="opus", target=resolved_targets["opus"]),
        ModelMappingRule(match="sonnet", target=resolved_targets["sonnet"]),
        ModelMappingRule(match="haiku", target=resolved_targets["haiku"]),
    ]


DEFAULT_GEMINI_MODEL_CARDS: list[ModelCard] = build_gemini_model_cards(
    additional_models=DEFAULT_GEMINI_ANTHROPIC_MODEL_TARGETS.values()
)

DEFAULT_GEMINI_MODEL_MAPPINGS: list[ModelMappingRule] = build_gemini_model_mappings()


__all__ = [
    "DEFAULT_GEMINI_ANTHROPIC_MODEL_TARGETS",
    "DEFAULT_GEMINI_FAST_MODEL",
    "DEFAULT_GEMINI_LIGHT_MODEL",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GEMINI_MODEL_CARDS",
    "DEFAULT_GEMINI_MODEL_MAPPINGS",
    "build_gemini_model_cards",
    "build_gemini_model_mappings",
]

