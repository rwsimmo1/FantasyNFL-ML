from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from fantasy_ml.scoring_rules import ScoringProvider

ScoringMode = Literal["PPR", "HALF_PPR", "STANDARD"]


@dataclass(frozen=True)
class AppConfig:
    fantasy_scoring: ScoringMode
    scoring_provider: ScoringProvider


def load_config(*, dotenv_path: str | None = None) -> AppConfig:
    """
    Load configuration from environment variables (optionally from a .env file).

    This function is small and unit-testable (tests can set env vars directly).
    """
    load_dotenv(dotenv_path=dotenv_path)

    scoring = os.getenv("FANTASY_SCORING", "PPR").strip().upper()
    if scoring not in ("PPR", "HALF_PPR", "STANDARD"):
        raise ValueError(
            "FANTASY_SCORING must be one of: PPR, HALF_PPR, STANDARD "
            f"(got {scoring!r})"
        )

    provider = os.getenv("SCORING_PROVIDER", "ESPN").strip().upper()
    if provider not in ("ESPN", "YAHOO"):
        raise ValueError(
            "SCORING_PROVIDER must be one of: ESPN, YAHOO "
            f"(got {provider!r})"
        )

    return AppConfig(  # type: ignore[return-value]
        fantasy_scoring=scoring,
        scoring_provider=provider,
    )