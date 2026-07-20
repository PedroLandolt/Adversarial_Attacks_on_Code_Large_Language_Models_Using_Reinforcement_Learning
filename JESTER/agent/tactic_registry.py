from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TacticRegistryEntry:
    tactic_id: str
    tactic_family: str
    environment_support: tuple[str, ...]
    renderer_binding: str
    taxonomy_category: str | None = None


_TACTIC_REGISTRY: tuple[TacticRegistryEntry, ...] = (
    TacticRegistryEntry(
        tactic_id="legacy_injection",
        tactic_family="injection",
        environment_support=("benchmark",),
        renderer_binding="prompt_injection",
        taxonomy_category="structural_logic",
    ),
    TacticRegistryEntry(
        tactic_id="legacy_output",
        tactic_family="output",
        environment_support=("benchmark",),
        renderer_binding="output_manipulation",
        taxonomy_category="obfuscation_noise",
    ),
    TacticRegistryEntry(
        tactic_id="legacy_semantic",
        tactic_family="semantic",
        environment_support=("benchmark",),
        renderer_binding="semantic_inconsistency",
        taxonomy_category="narrative_contextual",
    ),
    TacticRegistryEntry(
        tactic_id="legacy_cot",
        tactic_family="cot",
        environment_support=("benchmark",),
        renderer_binding="cot_poisoning",
        taxonomy_category="strategy_pacing",
    ),
    TacticRegistryEntry(
        tactic_id="taxonomy_roleplay",
        tactic_family="roleplay",
        environment_support=("benchmark",),
        renderer_binding="narrative_roleplay",
        taxonomy_category="narrative_contextual",
    ),
    TacticRegistryEntry(
        tactic_id="taxonomy_appeal_to_authority",
        tactic_family="appeal_to_authority",
        environment_support=("benchmark",),
        renderer_binding="pressure_authority",
        taxonomy_category="pressure_persuasion",
    ),
    TacticRegistryEntry(
        tactic_id="taxonomy_formatting_smuggling",
        tactic_family="formatting_smuggling",
        environment_support=("benchmark",),
        renderer_binding="structural_formatting_smuggling",
        taxonomy_category="structural_logic",
    ),
    TacticRegistryEntry(
        tactic_id="taxonomy_recursion_crescendo",
        tactic_family="recursion_crescendo",
        environment_support=("benchmark",),
        renderer_binding="strategy_recursion_crescendo",
        taxonomy_category="strategy_pacing",
    ),
    TacticRegistryEntry(
        tactic_id="taxonomy_crowding",
        tactic_family="crowding",
        environment_support=("benchmark",),
        renderer_binding="obfuscation_crowding",
        taxonomy_category="obfuscation_noise",
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
