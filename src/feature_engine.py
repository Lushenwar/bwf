"""
feature_engine.py — Build pre-match features from historical data.

CRITICAL RULE (Data Leakage Firewall 🔒):
For match at index N, we may ONLY use info from matches 0 … N-1.
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
    ROLLING_WINDOWS,
    ROUND_ORDINALS,
    IOC_TO_COUNTRY,
)
from src.glicko2 import Glicko2Tracker

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Helper: Parse scores
# ═══════════════════════════════════════════════════════════════════════════

def _parse_score_list(raw: object) -> list[tuple[int, int]] | None:
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

def _max_consecutive_from_scores(scores: list[tuple[int, int]], team_index: int) -> int:
    if not scores or len(scores) < 2:
        return 0
    max_streak = 0
    current_streak = 0
    for i in range(1, len(scores)):
        prev = scores[i - 1][team_index]
        curr = scores[i][team_index]
        if curr > prev:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak

# ═══════════════════════════════════════════════════════════════════════════
# Player history tracker
# ═══════════════════════════════════════════════════════════════════════════

class PlayerTracker:
    def __init__(self) -> None:
        self.elo: dict[tuple[str, str], float] = defaultdict(lambda: ELO_DEFAULT)
        self.record: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        self.recent: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.streaks: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.h2h: dict[tuple[str, frozenset], list[int]] = defaultdict(lambda: [0, 0])
        
        # Elite Tactical features
        self.pressure: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        self.closeness: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        self.max_streak_ever: dict[tuple[str, str], int] = defaultdict(int)

    # ── READ methods ───────────────────────────────────────────────────
    def get_elo(self, discipline: str, player: str) -> float:
        return self.elo[(discipline, player)]

    def get_win_rate(self, discipline: str, player: str) -> float:
        rec = self.record[(discipline, player)]
        if rec[1] == 0: return DEFAULT_WIN_RATE
        return rec[0] / rec[1]

    def get_matches_played(self, discipline: str, player: str) -> int:
        return self.record[(discipline, player)][1]

    def get_recent_form(self, discipline: str, player: str) -> float:
        history = self.recent[(discipline, player)]
        if not history: return DEFAULT_WIN_RATE
        window = history[-RECENT_FORM_WINDOW:]
        return sum(window) / len(window)

    def get_avg_streak(self, discipline: str, player: str) -> float:
        s = self.streaks[(discipline, player)]
        if not s: return 0.0
        return sum(s) / len(s)
        
    def get_streak_std(self, discipline: str, player: str) -> float:
        s = self.streaks[(discipline, player)]
        if len(s) < 2: return 0.0
        return float(np.std(s, ddof=1))

    def get_max_streak_ever(self, discipline: str, player: str) -> int:
        return self.max_streak_ever[(discipline, player)]

    def get_rolling_win_rate(self, discipline: str, player: str, window: int) -> float:
        history = self.recent[(discipline, player)]
        if not history: return DEFAULT_WIN_RATE
        recent = history[-window:]
        return sum(recent) / len(recent)

    def get_pressure_win_rate(self, discipline: str, player: str) -> float:
        rec = self.pressure[(discipline, player)]
        if rec[1] == 0: return DEFAULT_WIN_RATE
        return rec[0] / rec[1]

    def get_pressure_opportunities(self, discipline: str, player: str) -> int:
        return self.pressure[(discipline, player)][1]

    def get_close_game_pct(self, discipline: str, player: str) -> float:
        rec = self.closeness[(discipline, player)]
        if rec[1] == 0: return DEFAULT_WIN_RATE
        return rec[0] / rec[1]

    def get_h2h_win_rate(self, discipline: str, t1_key: frozenset, t2_key: frozenset) -> float:
        rec = self.h2h[(discipline, frozenset({t1_key, t2_key}))]
        if rec[1] == 0: return DEFAULT_WIN_RATE
        return rec[0] / rec[1]

    # ── WRITE methods ──────────────────────────────────────────────────
    def update_elo(self, discipline: str, winner: str, loser: str) -> None:
        r_w = self.elo[(discipline, winner)]
        r_l = self.elo[(discipline, loser)]
        e_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
        self.elo[(discipline, winner)] = r_w + ELO_K_FACTOR * (1.0 - e_w)
        self.elo[(discipline, loser)] = r_l + ELO_K_FACTOR * (0.0 - (1.0 - e_w))

    def update_record(self, discipline: str, player: str, won: bool) -> None:
        self.record[(discipline, player)][1] += 1
        if won: self.record[(discipline, player)][0] += 1

    def update_recent(self, discipline: str, player: str, won: bool) -> None:
        self.recent[(discipline, player)].append(1 if won else 0)

    def update_streak(self, discipline: str, player: str, streak_val: int) -> None:
        self.streaks[(discipline, player)].append(streak_val)
        if streak_val > self.max_streak_ever[(discipline, player)]:
            self.max_streak_ever[(discipline, player)] = streak_val

    def update_pressure(self, discipline: str, player: str, won: int, opps: int) -> None:
        self.pressure[(discipline, player)][0] += won
        self.pressure[(discipline, player)][1] += opps

    def update_closeness(self, discipline: str, player: str, was_close: bool) -> None:
        self.closeness[(discipline, player)][1] += 1
        if was_close: self.closeness[(discipline, player)][0] += 1

    def update_h2h(self, discipline: str, t1_key: frozenset, t2_key: frozenset, t1_won: bool) -> None:
        matchup = (discipline, frozenset({t1_key, t2_key}))
        self.h2h[matchup][1] += 1
        if t1_won: self.h2h[matchup][0] += 1

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_players(row: pd.Series, team: str) -> list[str]:
    p1 = row[f"team_{team}_p1"]
    p2 = row[f"team_{team}_p2"]
    players = [p1]
    if p2 is not None: players.append(p2)
    return players

def _team_key(players: list[str]) -> frozenset:
    return frozenset(players)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    tracker = PlayerTracker()
    glicko = Glicko2Tracker()
    epoch = pd.Timestamp("1970-01-01")
    n = len(df)
    
    # Feature Arrays
    feat = {
        "elo_team_one": np.zeros(n), "elo_team_two": np.zeros(n), "elo_diff": np.zeros(n),
        "elo_spread_team_one": np.zeros(n), "elo_spread_team_two": np.zeros(n),
        "win_rate_team_one": np.zeros(n), "win_rate_team_two": np.zeros(n), "win_rate_diff": np.zeros(n),
        "h2h_win_rate": np.zeros(n),
        "matches_played_team_one": np.zeros(n), "matches_played_team_two": np.zeros(n),
        "log_matches_team_one": np.zeros(n), "log_matches_team_two": np.zeros(n),
        "recent_form_team_one": np.zeros(n), "recent_form_team_two": np.zeros(n),
        "avg_point_streak_team_one": np.zeros(n), "avg_point_streak_team_two": np.zeros(n),
        "round_encoded": np.zeros(n),
        "is_home_team_one": np.zeros(n), "is_home_team_two": np.zeros(n),
        
        # Glicko
        "glicko_mu_team_one": np.zeros(n), "glicko_mu_team_two": np.zeros(n), "glicko_mu_diff": np.zeros(n),
        "glicko_rd_team_one": np.zeros(n), "glicko_rd_team_two": np.zeros(n), "glicko_rd_diff": np.zeros(n),
        "glicko_vol_team_one": np.zeros(n), "glicko_vol_team_two": np.zeros(n),
        
        # Tactical
        "pressure_wr_team_one": np.zeros(n), "pressure_wr_team_two": np.zeros(n),
        "pressure_opps_team_one": np.zeros(n), "pressure_opps_team_two": np.zeros(n),
        "close_game_pct_team_one": np.zeros(n), "close_game_pct_team_two": np.zeros(n),
        "max_streak_ever_team_one": np.zeros(n), "max_streak_ever_team_two": np.zeros(n),
        "streak_std_team_one": np.zeros(n), "streak_std_team_two": np.zeros(n),
    }
    
    for w in ROLLING_WINDOWS:
        feat[f"win_rate_last{w}_team_one"] = np.zeros(n)
        feat[f"win_rate_last{w}_team_two"] = np.zeros(n)
        
    logger.info("Building Elite features for %d matches...", n)
    
    for idx in range(n):
        row = df.iloc[idx]
        disc = row["discipline"]
        is_doubles = disc in DOUBLES_DISCIPLINES
        
        t1_p = _get_players(row, "one")
        t2_p = _get_players(row, "two")
        t1_k = _team_key(t1_p)
        t2_k = _team_key(t2_p)
        
        # 1. READ FEATURES
        # ELO
        t1_e = [tracker.get_elo(disc, p) for p in t1_p]
        t2_e = [tracker.get_elo(disc, p) for p in t2_p]
        feat["elo_team_one"][idx] = np.mean(t1_e)
        feat["elo_team_two"][idx] = np.mean(t2_e)
        feat["elo_diff"][idx] = feat["elo_team_one"][idx] - feat["elo_team_two"][idx]
        if is_doubles and len(t1_e)==2: feat["elo_spread_team_one"][idx] = abs(t1_e[0]-t1_e[1])
        if is_doubles and len(t2_e)==2: feat["elo_spread_team_two"][idx] = abs(t2_e[0]-t2_e[1])
        
        # General Stats
        feat["win_rate_team_one"][idx] = np.mean([tracker.get_win_rate(disc, p) for p in t1_p])
        feat["win_rate_team_two"][idx] = np.mean([tracker.get_win_rate(disc, p) for p in t2_p])
        feat["win_rate_diff"][idx] = feat["win_rate_team_one"][idx] - feat["win_rate_team_two"][idx]
        
        feat["h2h_win_rate"][idx] = tracker.get_h2h_win_rate(disc, t1_k, t2_k)
        
        mp1 = sum(tracker.get_matches_played(disc, p) for p in t1_p)
        mp2 = sum(tracker.get_matches_played(disc, p) for p in t2_p)
        feat["matches_played_team_one"][idx] = mp1
        feat["matches_played_team_two"][idx] = mp2
        feat["log_matches_team_one"][idx] = np.log1p(mp1)
        feat["log_matches_team_two"][idx] = np.log1p(mp2)
        
        feat["recent_form_team_one"][idx] = np.mean([tracker.get_recent_form(disc, p) for p in t1_p])
        feat["recent_form_team_two"][idx] = np.mean([tracker.get_recent_form(disc, p) for p in t2_p])
        
        feat["avg_point_streak_team_one"][idx] = np.mean([tracker.get_avg_streak(disc, p) for p in t1_p])
        feat["avg_point_streak_team_two"][idx] = np.mean([tracker.get_avg_streak(disc, p) for p in t2_p])
        
        feat["round_encoded"][idx] = ROUND_ORDINALS.get(row["round"], 5)
        
        # Rolling
        for w in ROLLING_WINDOWS:
            feat[f"win_rate_last{w}_team_one"][idx] = np.mean([tracker.get_rolling_win_rate(disc, p, w) for p in t1_p])
            feat[f"win_rate_last{w}_team_two"][idx] = np.mean([tracker.get_rolling_win_rate(disc, p, w) for p in t2_p])
            
        # Home Advantage
        ctry = row.get("country", "")
        def check_home(team_col):
            nats = []
            for sub in ["_p1_nationality", "_p2_nationality"]:
                val = row.get(f"team_{team_col}{sub}")
                if pd.notna(val): nats.append(str(val).strip())
            return float(any(IOC_TO_COUNTRY.get(n, "") == ctry for n in nats))
        feat["is_home_team_one"][idx] = check_home("one")
        feat["is_home_team_two"][idx] = check_home("two")
        
        # Glicko-2
        gm1 = [glicko.get_mu(disc, p) for p in t1_p]
        gm2 = [glicko.get_mu(disc, p) for p in t2_p]
        gr1 = [glicko.get_rd(disc, p) for p in t1_p]
        gr2 = [glicko.get_rd(disc, p) for p in t2_p]
        
        feat["glicko_mu_team_one"][idx] = np.mean(gm1)
        feat["glicko_mu_team_two"][idx] = np.mean(gm2)
        feat["glicko_mu_diff"][idx] = feat["glicko_mu_team_one"][idx] - feat["glicko_mu_team_two"][idx]
        feat["glicko_rd_team_one"][idx] = np.mean(gr1)
        feat["glicko_rd_team_two"][idx] = np.mean(gr2)
        feat["glicko_rd_diff"][idx] = feat["glicko_rd_team_one"][idx] - feat["glicko_rd_team_two"][idx]
        feat["glicko_vol_team_one"][idx] = np.mean([glicko.get_sigma(disc, p) for p in t1_p])
        feat["glicko_vol_team_two"][idx] = np.mean([glicko.get_sigma(disc, p) for p in t2_p])
        
        # Tactical
        feat["pressure_wr_team_one"][idx] = np.mean([tracker.get_pressure_win_rate(disc, p) for p in t1_p])
        feat["pressure_wr_team_two"][idx] = np.mean([tracker.get_pressure_win_rate(disc, p) for p in t2_p])
        feat["pressure_opps_team_one"][idx] = sum(tracker.get_pressure_opportunities(disc, p) for p in t1_p)
        feat["pressure_opps_team_two"][idx] = sum(tracker.get_pressure_opportunities(disc, p) for p in t2_p)
        
        feat["close_game_pct_team_one"][idx] = np.mean([tracker.get_close_game_pct(disc, p) for p in t1_p])
        feat["close_game_pct_team_two"][idx] = np.mean([tracker.get_close_game_pct(disc, p) for p in t2_p])
        
        feat["max_streak_ever_team_one"][idx] = max((tracker.get_max_streak_ever(disc, p) for p in t1_p), default=0)
        feat["max_streak_ever_team_two"][idx] = max((tracker.get_max_streak_ever(disc, p) for p in t2_p), default=0)
        feat["streak_std_team_one"][idx] = np.mean([tracker.get_streak_std(disc, p) for p in t1_p])
        feat["streak_std_team_two"][idx] = np.mean([tracker.get_streak_std(disc, p) for p in t2_p])
        
        # 2. UPDATE TRACKERS
        target = bool(row["target"] == 1)
        match_day = (row["date"] - epoch).days
        
        winners = t1_p if target else t2_p
        losers = t2_p if target else t1_p
        
        for w, l in zip(winners, losers):
            tracker.update_elo(disc, w, l)
            glicko.update(disc, w, l, match_day)
            
        for p in t1_p:
            tracker.update_record(disc, p, target)
            tracker.update_recent(disc, p, target)
        for p in t2_p:
            tracker.update_record(disc, p, not target)
            tracker.update_recent(disc, p, not target)
            
        tracker.update_h2h(disc, t1_k, t2_k, target)
        
        # Tactical Updates (Streaks/Pressure) - Parse games
        for g_idx in range(1, 4):
            key = f"game_{g_idx}_scores"
            if key not in row: continue
            sc = _parse_score_list(row.get(key))
            if not sc: continue
            
            # Streak
            s1 = _max_consecutive_from_scores(sc, 0)
            s2 = _max_consecutive_from_scores(sc, 1)
            for p in t1_p: tracker.update_streak(disc, p, s1)
            for p in t2_p: tracker.update_streak(disc, p, s2)
            
            # Pressure (15-15+)
            p1_won = 0; p1_opp = 0; p2_won = 0; p2_opp = 0
            for i in range(1, len(sc)):
                prev = sc[i-1]; curr = sc[i]
                if prev[0] >= 15 and prev[1] >= 15:
                    if curr[0] > prev[0]: p1_won +=1; p1_opp+=1; p2_opp+=1
                    elif curr[1] > prev[1]: p2_won+=1; p2_opp+=1; p1_opp+=1
            if p1_opp > 0 or p2_opp > 0:
                for p in t1_p: tracker.update_pressure(disc, p, p1_won, p1_opp)
                for p in t2_p: tracker.update_pressure(disc, p, p2_won, p2_opp)
                
            # Closeness (<=3 points)
            final = sc[-1]
            is_close = abs(final[0] - final[1]) <= 3
            for p in t1_p: tracker.update_closeness(disc, p, is_close)
            for p in t2_p: tracker.update_closeness(disc, p, is_close)
            
        if (idx + 1) % (n // 5) == 0:
            logger.info("...processed %d/%d (%.0f%%)", idx+1, n, 100*(idx+1)/n)

    # Assemble
    result = df.copy()
    for k, v in feat.items():
        result[k] = v
        
    result = pd.concat([result, pd.get_dummies(result["discipline"], prefix="disc")], axis=1)
    result = pd.concat([result, pd.get_dummies(result["tournament_type"], prefix="tourney")], axis=1)
    
    return result

NUMERIC_FEATURES = [
    "elo_team_one", "elo_team_two", "elo_diff", "elo_spread_team_one", "elo_spread_team_two",
    "win_rate_team_one", "win_rate_team_two", "win_rate_diff", "h2h_win_rate",
    "matches_played_team_one", "matches_played_team_two", "log_matches_team_one", "log_matches_team_two",
    "recent_form_team_one", "recent_form_team_two",
    "avg_point_streak_team_one", "avg_point_streak_team_two",
    "round_encoded",
    "is_home_team_one", "is_home_team_two",
    "glicko_mu_team_one", "glicko_mu_team_two", "glicko_mu_diff",
    "glicko_rd_team_one", "glicko_rd_team_two", "glicko_rd_diff",
    "glicko_vol_team_one", "glicko_vol_team_two",
    "pressure_wr_team_one", "pressure_wr_team_two", "pressure_opps_team_one", "pressure_opps_team_two",
    "close_game_pct_team_one", "close_game_pct_team_two",
    "max_streak_ever_team_one", "max_streak_ever_team_two", "streak_std_team_one", "streak_std_team_two",
] + [f"win_rate_last{w}_team_{t}" for w in ROLLING_WINDOWS for t in ["one", "two"]]

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = list(NUMERIC_FEATURES)
    cols.extend(sorted([c for c in df.columns if c.startswith("disc_")]))
    cols.extend(sorted([c for c in df.columns if c.startswith("tourney_")]))
    return cols
