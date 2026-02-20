"""
feature_engine.py — Build pre-match features from historical data.

CRITICAL RULE (Data Leakage Firewall 🔒):
==========================================
For match at index N (sorted by date), we may ONLY use information
from matches 0 … N-1.  Features are computed BEFORE we "see" the
outcome of match N, then we update our trackers AFTER the match.

How it works (explained simply):
================================
Imagine you're walking through a timeline of badminton matches.  For each
match you reach, you:
  1. Look BACKWARDS at everything that already happened.
  2. Write down numbers that describe each team's history (their ELO, how
     often they win, their recent streak, etc.).
  3. THEN you peek at who actually won, and update your "memory" so it's
     available for the NEXT match.

This ensures the model never "cheats" by seeing the future.

Features built:
===============
- ELO (discipline-isolated, per-player, averaged for doubles teams,
  with partner ELO spread for doubles)
- Historical win rate (discipline-specific)
- Head-to-head win rate
- Recent form (last 10 matches)
- Point streak persistence (from parsed game_*_scores of PAST matches)
- Match experience count
- Tournament type (one-hot)
- Round (ordinal)
- Discipline (one-hot)
"""

from __future__ import annotations

import ast
import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_WIN_RATE,
    DOUBLES_DISCIPLINES,
    ELO_DEFAULT,
    ELO_K_FACTOR,
    RECENT_FORM_WINDOW,
    ROUND_ORDINALS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Parse the game_*_scores string into a list of score tuples
# ═══════════════════════════════════════════════════════════════════════════

def _parse_score_list(raw: object) -> list[tuple[int, int]] | None:
    """Parse a stringified score list like "['0-0', '1-0', ...]".

    Returns a list of (team_one_score, team_two_score) tuples,
    or None if the input is NaN / unparseable.
    """
    if pd.isna(raw):
        return None
    try:
        items = ast.literal_eval(str(raw))
        result = []
        for s in items:
            parts = str(s).split("-")
            result.append((int(parts[0]), int(parts[1])))
        return result
    except (ValueError, SyntaxError, IndexError):
        return None


def _max_consecutive_from_scores(
    scores: list[tuple[int, int]], team_index: int
) -> int:
    """Compute the max consecutive points for a team from a score sequence.

    team_index: 0 → team_one, 1 → team_two.

    Why?  We need this because the `team_*_most_consecutive_points_game_*`
    columns exist in the CSV but we use scores from PAST matches only.
    """
    if not scores or len(scores) < 2:
        return 0

    max_streak = 0
    current_streak = 0

    for i in range(1, len(scores)):
        prev = scores[i - 1][team_index]
        curr = scores[i][team_index]
        if curr > prev:  # this team scored
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


# ═══════════════════════════════════════════════════════════════════════════
# Player history tracker — one instance stores all per-player state
# ═══════════════════════════════════════════════════════════════════════════

class PlayerTracker:
    """Tracks per-player, per-discipline historical stats.

    Every player gets a separate "bank account" per discipline.
    The key is always (discipline, player_name).
    """

    def __init__(self) -> None:
        # ELO rating per (discipline, player)
        self.elo: dict[tuple[str, str], float] = defaultdict(
            lambda: ELO_DEFAULT
        )

        # Win/loss record per (discipline, player): [wins, total]
        self.record: dict[tuple[str, str], list[int]] = defaultdict(
            lambda: [0, 0]
        )

        # Recent results per (discipline, player): list of 1/0  (last N)
        self.recent: dict[tuple[str, str], list[int]] = defaultdict(list)

        # Point streak history per (discipline, player): list of max_streak values
        self.streaks: dict[tuple[str, str], list[int]] = defaultdict(list)

        # Head-to-head: (discipline, frozenset({p1_key, p2_key})) → [team_one_wins, total]
        # For singles p_key = player name; for doubles p_key = frozenset of player names
        self.h2h: dict[tuple[str, frozenset], list[int]] = defaultdict(
            lambda: [0, 0]
        )

    # ── Read methods (pre-match) ───────────────────────────────────────

    def get_elo(self, discipline: str, player: str) -> float:
        return self.elo[(discipline, player)]

    def get_win_rate(self, discipline: str, player: str) -> float:
        rec = self.record[(discipline, player)]
        if rec[1] == 0:
            return DEFAULT_WIN_RATE
        return rec[0] / rec[1]

    def get_matches_played(self, discipline: str, player: str) -> int:
        return self.record[(discipline, player)][1]

    def get_recent_form(self, discipline: str, player: str) -> float:
        history = self.recent[(discipline, player)]
        if not history:
            return DEFAULT_WIN_RATE
        window = history[-RECENT_FORM_WINDOW:]
        return sum(window) / len(window)

    def get_avg_streak(self, discipline: str, player: str) -> float:
        s = self.streaks[(discipline, player)]
        if not s:
            return 0.0
        return sum(s) / len(s)

    def get_h2h_win_rate(
        self, discipline: str, team_one_key: frozenset, team_two_key: frozenset
    ) -> float:
        """Return team_one's historical win rate against team_two."""
        matchup = (discipline, frozenset({team_one_key, team_two_key}))
        rec = self.h2h[matchup]
        if rec[1] == 0:
            return DEFAULT_WIN_RATE
        return rec[0] / rec[1]

    # ── Write methods (post-match) ─────────────────────────────────────

    def update_elo(
        self,
        discipline: str,
        winner: str,
        loser: str,
    ) -> None:
        """Update ELO for two players after a match result."""
        w_key = (discipline, winner)
        l_key = (discipline, loser)

        r_w = self.elo[w_key]
        r_l = self.elo[l_key]

        # Expected scores (logistic curve)
        e_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
        e_l = 1.0 - e_w

        self.elo[w_key] = r_w + ELO_K_FACTOR * (1.0 - e_w)
        self.elo[l_key] = r_l + ELO_K_FACTOR * (0.0 - e_l)

    def update_record(
        self, discipline: str, player: str, won: bool
    ) -> None:
        key = (discipline, player)
        self.record[key][1] += 1  # total matches
        if won:
            self.record[key][0] += 1  # wins

    def update_recent(
        self, discipline: str, player: str, won: bool
    ) -> None:
        self.recent[(discipline, player)].append(1 if won else 0)

    def update_streak(
        self, discipline: str, player: str, streak_val: int
    ) -> None:
        self.streaks[(discipline, player)].append(streak_val)

    def update_h2h(
        self,
        discipline: str,
        team_one_key: frozenset,
        team_two_key: frozenset,
        team_one_won: bool,
    ) -> None:
        matchup = (discipline, frozenset({team_one_key, team_two_key}))
        self.h2h[matchup][1] += 1
        if team_one_won:
            self.h2h[matchup][0] += 1


# ═══════════════════════════════════════════════════════════════════════════
# Helpers for extracting team information from a row
# ═══════════════════════════════════════════════════════════════════════════

def _get_players(row: pd.Series, team: str) -> list[str]:
    """Return list of player names for a team (1 for singles, 2 for doubles).

    team: 'one' or 'two'.
    """
    p1 = row[f"team_{team}_p1"]
    p2 = row[f"team_{team}_p2"]
    players = [p1]
    if p2 is not None:
        players.append(p2)
    return players


def _team_key(players: list[str]) -> frozenset:
    """Create a hashable key for a team (frozenset of player names)."""
    return frozenset(players)


# ═══════════════════════════════════════════════════════════════════════════
# Main feature-building function
# ═══════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Walk through the DataFrame chronologically and build features.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_and_clean().  Must be sorted by date.

    Returns
    -------
    pd.DataFrame
        A copy of the input with new feature columns appended.
        The post-match columns from the original CSV remain for reference
        but should NEVER be used as model inputs.
    """
    assert df["date"].is_monotonic_increasing, (
        "DataFrame must be sorted by date! Temporal integrity violated."
    )

    tracker = PlayerTracker()

    # Pre-allocate feature arrays (much faster than row-by-row DataFrame writes)
    n = len(df)
    feat = {
        "elo_team_one": np.zeros(n),
        "elo_team_two": np.zeros(n),
        "elo_diff": np.zeros(n),
        "elo_spread_team_one": np.zeros(n),
        "elo_spread_team_two": np.zeros(n),
        "win_rate_team_one": np.zeros(n),
        "win_rate_team_two": np.zeros(n),
        "h2h_win_rate": np.zeros(n),
        "matches_played_team_one": np.zeros(n),
        "matches_played_team_two": np.zeros(n),
        "recent_form_team_one": np.zeros(n),
        "recent_form_team_two": np.zeros(n),
        "avg_point_streak_team_one": np.zeros(n),
        "avg_point_streak_team_two": np.zeros(n),
        "round_encoded": np.zeros(n),
    }

    logger.info("Building features for %d matches…", n)

    for idx in range(n):
        row = df.iloc[idx]
        disc = row["discipline"]
        is_doubles = disc in DOUBLES_DISCIPLINES

        # --- Identify players ---
        t1_players = _get_players(row, "one")
        t2_players = _get_players(row, "two")
        t1_key = _team_key(t1_players)
        t2_key = _team_key(t2_players)

        # ── STEP 1: READ pre-match features (BEFORE seeing outcome) ────

        # ELO (average for doubles, single for singles)
        t1_elos = [tracker.get_elo(disc, p) for p in t1_players]
        t2_elos = [tracker.get_elo(disc, p) for p in t2_players]

        elo_t1 = sum(t1_elos) / len(t1_elos)
        elo_t2 = sum(t2_elos) / len(t2_elos)

        feat["elo_team_one"][idx] = elo_t1
        feat["elo_team_two"][idx] = elo_t2
        feat["elo_diff"][idx] = elo_t1 - elo_t2

        # ELO spread (partner gap — 0 for singles)
        if is_doubles and len(t1_elos) == 2:
            feat["elo_spread_team_one"][idx] = abs(t1_elos[0] - t1_elos[1])
        if is_doubles and len(t2_elos) == 2:
            feat["elo_spread_team_two"][idx] = abs(t2_elos[0] - t2_elos[1])

        # Win rate (average across team members)
        feat["win_rate_team_one"][idx] = np.mean(
            [tracker.get_win_rate(disc, p) for p in t1_players]
        )
        feat["win_rate_team_two"][idx] = np.mean(
            [tracker.get_win_rate(disc, p) for p in t2_players]
        )

        # Head-to-head
        feat["h2h_win_rate"][idx] = tracker.get_h2h_win_rate(disc, t1_key, t2_key)

        # Matches played (sum across team members)
        feat["matches_played_team_one"][idx] = sum(
            tracker.get_matches_played(disc, p) for p in t1_players
        )
        feat["matches_played_team_two"][idx] = sum(
            tracker.get_matches_played(disc, p) for p in t2_players
        )

        # Recent form
        feat["recent_form_team_one"][idx] = np.mean(
            [tracker.get_recent_form(disc, p) for p in t1_players]
        )
        feat["recent_form_team_two"][idx] = np.mean(
            [tracker.get_recent_form(disc, p) for p in t2_players]
        )

        # Point streak persistence
        feat["avg_point_streak_team_one"][idx] = np.mean(
            [tracker.get_avg_streak(disc, p) for p in t1_players]
        )
        feat["avg_point_streak_team_two"][idx] = np.mean(
            [tracker.get_avg_streak(disc, p) for p in t2_players]
        )

        # Round ordinal
        feat["round_encoded"][idx] = ROUND_ORDINALS.get(row["round"], 5)

        # ── STEP 2: PEEK at the outcome ────────────────────────────────

        team_one_won = bool(row["target"] == 1)

        # ── STEP 3: UPDATE trackers (so next match can see this result) ─

        # ELO updates — per individual player
        for wp, lp in zip(
            t1_players if team_one_won else t2_players,
            t2_players if team_one_won else t1_players,
        ):
            tracker.update_elo(disc, wp, lp)

        # Record + recent form
        for p in t1_players:
            tracker.update_record(disc, p, team_one_won)
            tracker.update_recent(disc, p, team_one_won)
        for p in t2_players:
            tracker.update_record(disc, p, not team_one_won)
            tracker.update_recent(disc, p, not team_one_won)

        # Head-to-head
        tracker.update_h2h(disc, t1_key, t2_key, team_one_won)

        # Point streaks from this match's score sequences (for FUTURE matches)
        for game_col, team_idx_label in [
            ("game_1_scores", None),
            ("game_2_scores", None),
            ("game_3_scores", None),
        ]:
            if game_col not in df.columns:
                continue
            scores = _parse_score_list(row.get(game_col))
            if scores is None:
                continue
            t1_streak = _max_consecutive_from_scores(scores, 0)
            t2_streak = _max_consecutive_from_scores(scores, 1)
            for p in t1_players:
                tracker.update_streak(disc, p, t1_streak)
            for p in t2_players:
                tracker.update_streak(disc, p, t2_streak)

        # Progress logging (every 20%)
        if (idx + 1) % (n // 5) == 0:
            logger.info("  … processed %d / %d matches (%.0f%%)",
                        idx + 1, n, 100 * (idx + 1) / n)

    # ── Assemble feature DataFrame ─────────────────────────────────────

    result = df.copy()
    for col_name, values in feat.items():
        result[col_name] = values

    # One-hot encode discipline
    disc_dummies = pd.get_dummies(result["discipline"], prefix="disc")
    result = pd.concat([result, disc_dummies], axis=1)

    # One-hot encode tournament_type
    tt_dummies = pd.get_dummies(result["tournament_type"], prefix="tourney")
    result = pd.concat([result, tt_dummies], axis=1)

    logger.info("Feature engineering complete. Shape: %s", result.shape)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Get the list of feature column names (for the model)
# ═══════════════════════════════════════════════════════════════════════════

NUMERIC_FEATURES = [
    "elo_team_one",
    "elo_team_two",
    "elo_diff",
    "elo_spread_team_one",
    "elo_spread_team_two",
    "win_rate_team_one",
    "win_rate_team_two",
    "h2h_win_rate",
    "matches_played_team_one",
    "matches_played_team_two",
    "recent_form_team_one",
    "recent_form_team_two",
    "avg_point_streak_team_one",
    "avg_point_streak_team_two",
    "round_encoded",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns to use as model inputs.

    This combines the numeric features with the one-hot encoded columns
    that were generated dynamically.
    """
    feature_cols = list(NUMERIC_FEATURES)

    # Add discipline one-hot columns
    disc_cols = [c for c in df.columns if c.startswith("disc_")]
    feature_cols.extend(sorted(disc_cols))

    # Add tournament type one-hot columns
    tt_cols = [c for c in df.columns if c.startswith("tourney_")]
    feature_cols.extend(sorted(tt_cols))

    return feature_cols


# ── Quick smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")
    from src.data_loader import load_and_clean

    df = load_and_clean()
    featured = build_features(df)

    feat_cols = get_feature_columns(featured)
    print(f"\n--- Feature columns ({len(feat_cols)}) ---")
    for c in feat_cols:
        print(f"  {c}")

    print(f"\n--- Sample features (first 5 rows) ---")
    print(featured[feat_cols].head().to_string())
    print(f"\n--- Feature stats ---")
    print(featured[feat_cols].describe().to_string())
