from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TacticRegistryEntry:
    tactic_id: str
    tactic_family: str
    environment_support: tuple[str, ...]
    renderer_binding: str


_TACTIC_REGISTRY: tuple[TacticRegistryEntry, ...] = (
    TacticRegistryEntry(
        tactic_id="legacy_injection",
        tactic_family="injection",
        environment_support=("benchmark", "gitea"),
        renderer_binding="prompt_injection",
    ),
    TacticRegistryEntry(
        tactic_id="legacy_output",
        tactic_family="output",
        environment_support=("benchmark", "gitea"),
        renderer_binding="output_manipulation",
    ),
    TacticRegistryEntry(
        tactic_id="legacy_semantic",
        tactic_family="semantic",
        environment_support=("benchmark", "gitea"),
        renderer_binding="semantic_inconsistency",
    ),
    TacticRegistryEntry(
        tactic_id="legacy_cot",
        tactic_family="cot",
        environment_support=("benchmark", "gitea"),
        renderer_binding="cot_poisoning",
    ),
)


def get_tactic_registry(environment: str | None = None) -> list[TacticRegistryEntry]:
    if environment is None:
        return list(_TACTIC_REGISTRY)
    normalized_environment = str(environment).strip().lower()
    return [
        entry
        for entry in _TACTIC_REGISTRY
        if normalized_environment in entry.environment_support
    ]


def get_tactic_entry(tactic_family_or_id: str, environment: str | None = None) -> TacticRegistryEntry:
    normalized = str(tactic_family_or_id).strip().lower()
    for entry in get_tactic_registry(environment):
        if normalized in {entry.tactic_id, entry.tactic_family}:
            return entry
    raise ValueError(f"Unknown tactic/action: {tactic_family_or_id}")


def get_supported_tactic_families(environment: str | None = None) -> list[str]:
    return [entry.tactic_family for entry in get_tactic_registry(environment)]
