"""
predict.py — Inference module for Shuttle-X.

What this module does (explained simply):
=========================================
After we've trained the model (train.py), this module loads the saved
model and lets you ask: "If Player A faces Player B in a Super 500
Quarter Final in Men's Singles, who's more likely to win?"

It returns a probability — e.g., "Player A has a 73% chance of winning."

IMPORTANT — Input Sanitization:
================================
This module validates all inputs before running predictions:
- Player names are stripped and title-cased.
- Discipline must be one of: MS, WS, MD, WD, XD.
- Tournament type and round must be recognized values.
- Unknown values get safe defaults rather than crashing.

Usage:
    from src.predict import ShuttleXPredictor
    predictor = ShuttleXPredictor()
    result = predictor.predict(
        discipline="MS",
        team_one_players=["Kento Momota"],
        team_two_players=["Viktor Axelsen"],
        tournament_type="HSBC BWF World Tour Super 1000",
        round_name="Final",
    )
    print(result)  # {'team_one_win_prob': 0.73, 'team_two_win_prob': 0.27, ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import (
    MODELS_DIR,
    ROUND_ORDINALS,
    VALID_DISCIPLINES,
    DOUBLES_DISCIPLINES,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Input sanitization
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize_player_name(name: str) -> str:
    """Clean a player name: strip whitespace, title-case."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Invalid player name: {name!r}")
    return name.strip().title()


def _validate_discipline(discipline: str) -> str:
    """Ensure discipline is one of the allowed values."""
    discipline = str(discipline).strip().upper()
    if discipline not in VALID_DISCIPLINES:
        raise ValueError(
            f"Invalid discipline: {discipline!r}. "
            f"Must be one of: {sorted(VALID_DISCIPLINES)}"
        )
    return discipline


def _validate_players(
    discipline: str, players: list[str]
) -> list[str]:
    """Validate player list length matches the discipline type."""
    sanitized = [_sanitize_player_name(p) for p in players]

    if discipline in DOUBLES_DISCIPLINES:
        if len(sanitized) < 1 or len(sanitized) > 2:
            raise ValueError(
                f"Doubles discipline {discipline!r} requires 1-2 players, "
                f"got {len(sanitized)}"
            )
    else:
        if len(sanitized) != 1:
            raise ValueError(
                f"Singles discipline {discipline!r} requires exactly 1 player, "
                f"got {len(sanitized)}"
            )
    return sanitized


# ═══════════════════════════════════════════════════════════════════════════
# Predictor class
# ═══════════════════════════════════════════════════════════════════════════

class ShuttleXPredictor:
    """Load a trained model and make match predictions."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        model_path = model_path or MODELS_DIR / "shuttle_x_model.json"
        metadata_path = metadata_path or MODELS_DIR / "shuttle_x_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run `python -m src.train` first."
            )

        # Load model
        self.model = XGBClassifier()
        self.model.load_model(str(model_path))
        logger.info("Model loaded from %s", model_path)

        # Load metadata (feature columns, etc.)
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self.feature_columns = self.metadata["feature_columns"]
        logger.info("Metadata loaded — %d feature columns", len(self.feature_columns))

    def predict(
        self,
        discipline: str,
        team_one_players: list[str],
        team_two_players: list[str],
        tournament_type: str = "HSBC BWF World Tour Super 500",
        round_name: str = "Quarter final",
        elo_team_one: float = 1500.0,
        elo_team_two: float = 1500.0,
        win_rate_team_one: float = 0.5,
        win_rate_team_two: float = 0.5,
        h2h_win_rate: float = 0.5,
        matches_played_team_one: int = 0,
        matches_played_team_two: int = 0,
        recent_form_team_one: float = 0.5,
        recent_form_team_two: float = 0.5,
        avg_point_streak_team_one: float = 3.0,
        avg_point_streak_team_two: float = 3.0,
        elo_spread_team_one: float = 0.0,
        elo_spread_team_two: float = 0.0,
    ) -> dict:
        """Predict the outcome of a match.

        Parameters
        ----------
        discipline : str
            One of MS, WS, MD, WD, XD.
        team_one_players / team_two_players : list[str]
            Player name(s). 1 for singles, 1-2 for doubles.
        tournament_type : str
            Tournament tier name.
        round_name : str
            Round name (e.g., "Quarter final").
        elo_* / win_rate_* / etc. : float
            Pre-computed stats for the teams. In production, these come
            from the live ELO tracker; for ad-hoc predictions, you can
            supply custom values.

        Returns
        -------
        dict with keys:
            team_one_win_prob, team_two_win_prob, prediction, team_one_players,
            team_two_players, discipline
        """
        # ── Sanitize inputs ────────────────────────────────────────────
        discipline = _validate_discipline(discipline)
        team_one_players = _validate_players(discipline, team_one_players)
        team_two_players = _validate_players(discipline, team_two_players)

        # ── Build feature row ──────────────────────────────────────────
        row: dict[str, float] = {
            "elo_team_one": float(elo_team_one),
            "elo_team_two": float(elo_team_two),
            "elo_diff": float(elo_team_one - elo_team_two),
            "elo_spread_team_one": float(elo_spread_team_one),
            "elo_spread_team_two": float(elo_spread_team_two),
            "win_rate_team_one": float(win_rate_team_one),
            "win_rate_team_two": float(win_rate_team_two),
            "h2h_win_rate": float(h2h_win_rate),
            "matches_played_team_one": float(matches_played_team_one),
            "matches_played_team_two": float(matches_played_team_two),
            "recent_form_team_one": float(recent_form_team_one),
            "recent_form_team_two": float(recent_form_team_two),
            "avg_point_streak_team_one": float(avg_point_streak_team_one),
            "avg_point_streak_team_two": float(avg_point_streak_team_two),
            "round_encoded": float(ROUND_ORDINALS.get(round_name, 5)),
        }

        # Discipline one-hot
        for disc_val in ["MD", "MS", "WD", "WS", "XD"]:
            col = f"disc_{disc_val}"
            row[col] = 1.0 if discipline == disc_val else 0.0

        # Tournament type one-hot
        known_tourneys = [c for c in self.feature_columns if c.startswith("tourney_")]
        for t_col in known_tourneys:
            t_name = t_col.replace("tourney_", "", 1)
            row[t_col] = 1.0 if tournament_type == t_name else 0.0

        # ── Build DataFrame in correct column order ────────────────────
        X = pd.DataFrame([row])[self.feature_columns]

        # ── Predict ────────────────────────────────────────────────────
        prob = float(self.model.predict_proba(X)[0, 1])

        return {
            "team_one_win_prob": round(prob, 4),
            "team_two_win_prob": round(1 - prob, 4),
            "prediction": "Team One Wins" if prob >= 0.5 else "Team Two Wins",
            "team_one_players": team_one_players,
            "team_two_players": team_two_players,
            "discipline": discipline,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Quick demo
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")

    predictor = ShuttleXPredictor()

    # Demo: a high-ELO player vs a low-ELO player in MS
    result = predictor.predict(
        discipline="MS",
        team_one_players=["Kento Momota"],
        team_two_players=["Unknown Player"],
        tournament_type="HSBC BWF World Tour Super 1000",
        round_name="Final",
        elo_team_one=1900,
        elo_team_two=1400,
        win_rate_team_one=0.85,
        win_rate_team_two=0.40,
        matches_played_team_one=90,
        matches_played_team_two=5,
        recent_form_team_one=0.9,
        recent_form_team_two=0.3,
    )

    print("\n--- Prediction ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
