"""
data_loader.py — Load, clean, and unify the five BWF discipline CSVs.

What this module does (explained simply):
=========================================
1. Reads all five CSV files (ms, ws, md, wd, xd) from the data/ folder.
2. Normalises the player columns so that Singles and Doubles files share
   the same column names.  Singles files have one player per team;
   Doubles files have two.  We standardise to:
       team_one_p1, team_one_p2, team_two_p1, team_two_p2
   For Singles, p2 is set to None.
3. Drops "retirement" rows (winner == 0) — these are unpredictable.
4. Parses the date column (DD-MM-YYYY → proper datetime).
5. Sorts everything by date so the feature engine can walk forward in time.
6. Remaps the target: winner 1 → 1 (team one wins), winner 2 → 0 (team two wins).

⚠️  This module touches ONLY the raw data.  It does NOT create any features.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.config import (
    CSV_FILES,
    DATA_DIR,
    DATE_FORMAT,
    DOUBLES_DISCIPLINES,
    SINGLES_DISCIPLINES,
)

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────


def _sanitize_name(name: Optional[str]) -> Optional[str]:
    """Strip whitespace and normalise case for player names.

    Why?  The same person might appear as 'Kento Momota', 'kento momota',
    or ' Kento Momota '.  We force Title Case and strip spaces so that
    the ELO tracker doesn't create duplicate "bank accounts" for the
    same player.
    """
    if name is None or (isinstance(name, float)):  # NaN check
        return None
    name = str(name).strip()
    if not name:
        return None
    return name.title()


def _load_single_csv(discipline: str, filename: str) -> pd.DataFrame:
    """Load one discipline CSV file and normalise its player columns."""
    filepath = DATA_DIR / filename
    logger.info("Loading %s from %s", discipline, filepath)
    df = pd.read_csv(filepath)

    # --- Normalise player columns ---
    if discipline in SINGLES_DISCIPLINES:
        # Singles: one player per team
        df = df.rename(columns={
            "team_one_players": "team_one_p1",
            "team_two_players": "team_two_p1",
            "team_one_nationalities": "team_one_p1_nationality",
            "team_two_nationalities": "team_two_p1_nationality",
        })
        df["team_one_p2"] = None
        df["team_two_p2"] = None
        df["team_one_p2_nationality"] = None
        df["team_two_p2_nationality"] = None

    elif discipline in DOUBLES_DISCIPLINES:
        # Doubles: two players per team
        df = df.rename(columns={
            "team_one_player_one": "team_one_p1",
            "team_one_player_two": "team_one_p2",
            "team_two_player_one": "team_two_p1",
            "team_two_player_two": "team_two_p2",
            "team_one_player_one_nationality": "team_one_p1_nationality",
            "team_one_player_two_nationality": "team_one_p2_nationality",
            "team_two_player_one_nationality": "team_two_p1_nationality",
            "team_two_player_two_nationality": "team_two_p2_nationality",
        })
    else:
        raise ValueError(f"Unknown discipline: {discipline!r}")

    return df


# ── public API ─────────────────────────────────────────────────────────────


def load_and_clean() -> pd.DataFrame:
    """Load all five CSVs, clean, unify, and return a single DataFrame.

    Returns
    -------
    pd.DataFrame
        Sorted chronologically.  Columns include:
        - date (datetime64)
        - discipline (str)
        - winner (int): 1 → team one won, 0 → team two won
        - team_one_p1, team_one_p2, team_two_p1, team_two_p2
        - tournament, city, country, tournament_type, round
        - All original post-match columns are RETAINED in the DataFrame
          (they'll be used by the feature engine to derive HISTORICAL stats
          from past matches only).  They are never used as direct features.
    """
    frames: list[pd.DataFrame] = []

    for discipline, filename in CSV_FILES.items():
        df = _load_single_csv(discipline, filename)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %d rows", len(combined))

    # ── 1. Drop retirements (winner == 0) ──────────────────────────────
    before = len(combined)
    combined = combined[combined["winner"] != 0].copy()
    dropped = before - len(combined)
    logger.info("Dropped %d retirement rows (winner=0). Remaining: %d",
                dropped, len(combined))

    # ── 2. Parse dates ─────────────────────────────────────────────────
    combined["date"] = pd.to_datetime(combined["date"], format=DATE_FORMAT)

    # ── 3. Sort by date (temporal integrity!) ──────────────────────────
    combined = combined.sort_values("date", kind="mergesort").reset_index(drop=True)
    logger.info("Date range: %s → %s",
                combined["date"].min().date(),
                combined["date"].max().date())

    # ── 4. Sanitise player names ───────────────────────────────────────
    for col in ["team_one_p1", "team_one_p2", "team_two_p1", "team_two_p2"]:
        combined[col] = combined[col].apply(_sanitize_name)

    # ── 5. Remap target: winner 1 → 1, winner 2 → 0 ──────────────────
    combined["target"] = (combined["winner"] == 1).astype(int)

    logger.info("Final dataset ready: %d rows, %d columns",
                len(combined), len(combined.columns))

    return combined


# ── Quick smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")
    df = load_and_clean()
    print("\n--- Sample output (first 5 rows, key columns) ---")
    cols = ["date", "discipline", "target", "tournament_type", "round",
            "team_one_p1", "team_one_p2", "team_two_p1", "team_two_p2"]
    print(df[cols].head().to_string())
    print(f"\nTarget distribution:\n{df['target'].value_counts().to_string()}")
    print(f"\nDiscipline distribution:\n{df['discipline'].value_counts().to_string()}")
