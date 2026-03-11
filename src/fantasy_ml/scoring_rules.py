from __future__ import annotations

from typing import Dict, Literal

ScoringMode = Literal["PPR", "HALF_PPR", "STANDARD"]
ScoringProvider = Literal["ESPN", "YAHOO"]

_BASE_RULES: Dict[str, float] = {
    "pass_yd": 1.0 / 25.0,
    "pass_td": 4.0,
    "pass_int": -2.0,
    "rush_yd": 1.0 / 10.0,
    "rush_td": 6.0,
    "rec_yd": 1.0 / 10.0,
    "rec_td": 6.0,
    "fumble_lost": -2.0,
    "def_sack": 1.0,
    "def_int": 2.0,
    "def_fumble_recovery": 2.0,
    "def_td": 6.0,
}

SCORING_RULES: Dict[ScoringProvider, Dict[ScoringMode, Dict[str, float]]] = {
    "ESPN": {
        "PPR": {**_BASE_RULES, "rec": 1.0},
        "HALF_PPR": {**_BASE_RULES, "rec": 0.5},
        "STANDARD": {**_BASE_RULES, "rec": 0.0},
    },
    "YAHOO": {
        "PPR": {**_BASE_RULES, "rec": 1.0},
        "HALF_PPR": {**_BASE_RULES, "rec": 0.5},
        "STANDARD": {**_BASE_RULES, "rec": 0.0},
    },
}


def normalize_scoring_mode(mode: str) -> ScoringMode:
    """Normalize and validate scoring mode."""
    normalized_mode = mode.strip().upper()
    if normalized_mode in ("PPR", "HALF_PPR", "STANDARD"):
        return normalized_mode  # type: ignore[return-value]

    raise ValueError(
        "scoring mode must be one of: PPR, HALF_PPR, STANDARD "
        f"(got {mode!r})"
    )


def normalize_scoring_provider(provider: str) -> ScoringProvider:
    """Normalize and validate scoring provider."""
    normalized_provider = provider.strip().upper()
    if normalized_provider in ("ESPN", "YAHOO"):
        return normalized_provider  # type: ignore[return-value]

    raise ValueError(
        "scoring provider must be one of: ESPN, YAHOO "
        f"(got {provider!r})"
    )


def get_scoring_rules(mode: str, provider: str = "ESPN") -> Dict[str, float]:
    """Return scoring-rule weights for the given mode and provider."""
    normalized_provider = normalize_scoring_provider(provider)
    normalized_mode = normalize_scoring_mode(mode)
    return SCORING_RULES[normalized_provider][normalized_mode]
