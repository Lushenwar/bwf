import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.special import expit
import math
import json
import os

from src.config import MODELS_DIR, ROLLING_WINDOWS, ROUND_ORDINALS, DOUBLES_DISCIPLINES, IOC_TO_COUNTRY
from src.data_loader import load_and_clean, _sanitize_name
from src.feature_engine import PlayerTracker, _get_players, _team_key, _parse_score_list, _max_consecutive_from_scores
from src.glicko2 import Glicko2Tracker

logger = logging.getLogger("BATCH")

def get_replay_state(df: pd.DataFrame):
    """Replay history to build the current state of trackers."""
    tracker = PlayerTracker()
    glicko = Glicko2Tracker()
    epoch = pd.Timestamp("1970-01-01")
    
    logger.info(f"Replaying {len(df)} matches to build state...")
    
    for idx, row in df.iterrows():
        disc = row["discipline"]
        target = bool(row["target"] == 1)
        match_day = (row["date"] - epoch).days
        
        t1_p = _get_players(row, "one")
        t2_p = _get_players(row, "two")
        t1_k = _team_key(t1_p)
        t2_k = _team_key(t2_p)
        
        winners = t1_p if target else t2_p
        losers = t2_p if target else t1_p
        
        # Update Trackers
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
        
        # Tactical Updates
        for g_idx in range(1, 4):
            key = f"game_{g_idx}_scores"
            if key not in row: continue
            sc = _parse_score_list(row.get(key))
            if not sc: continue
            
            s1 = _max_consecutive_from_scores(sc, 0)
            s2 = _max_consecutive_from_scores(sc, 1)
            for p in t1_p: tracker.update_streak(disc, p, s1)
            for p in t2_p: tracker.update_streak(disc, p, s2)
            
            p1_won = 0; p1_opp = 0; p2_won = 0; p2_opp = 0
            for i in range(1, len(sc)):
                prev = sc[i-1]; curr = sc[i]
                if prev[0] >= 15 and prev[1] >= 15:
                    if curr[0] > prev[0]: p1_won +=1; p1_opp+=1; p2_opp+=1
                    elif curr[1] > prev[1]: p2_won+=1; p2_opp+=1; p1_opp+=1
            if p1_opp > 0:
                for p in t1_p: tracker.update_pressure(disc, p, p1_won, p1_opp)
                for p in t2_p: tracker.update_pressure(disc, p, p2_won, p2_opp)

            final = sc[-1]
            is_close = abs(final[0] - final[1]) <= 3
            for p in t1_p: tracker.update_closeness(disc, p, is_close)
            for p in t2_p: tracker.update_closeness(disc, p, is_close)
            
    return tracker, glicko

def load_artifacts():
    """Load the trained XGBoost model and metadata."""
    try:
        xgb_model = xgb.XGBClassifier()
        # Load the JSON model which is compatible with XGBClassifier or Booster
        xgb_model.load_model(MODELS_DIR / "shuttle_x_model.json")
        
        with open(MODELS_DIR / "shuttle_x_metadata.json", "r") as f:
            meta = json.load(f)
            
        return xgb_model, meta
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        logger.warning("Returning dummy models for testing.")
        # Minimal dummy
        return None, {"feature_columns": []}

class BatchPredictor:
    def __init__(self):
        self.tracker = None
        self.glicko = None
        self.xgb_model = None
        self.meta = None
        self.epoch = pd.Timestamp("1970-01-01")

    def load(self):
        print("Loading data and replaying history (this takes ~10 seconds)...")
        df = load_and_clean()
        self.tracker, self.glicko = get_replay_state(df)
        self.xgb_model, self.meta = load_artifacts()
        
        if self.xgb_model is None:
            raise ValueError("Failed to load model from disk! Please run run_training.py")
            
        print("Ready for predictions.")

    def get_known_players(self, discipline):
        """Return sorted list of all players seen in history for this discipline."""
        if not self.tracker:
            return []
        players = set()
        for k in self.tracker.elo.keys():
            if k[0] == discipline:
                players.add(k[1])
        return sorted(list(players))

    def predict(self, date_str, discipline, t1_p1_name, t1_p1_nat, t1_p2_name, t1_p2_nat, 
                t2_p1_name, t2_p1_nat, t2_p2_name, t2_p2_nat, tournament, country, round_name):
        
        if self.xgb_model is None:
             raise ValueError("Models not loaded! Call load() first.")

        match_date = pd.to_datetime(date_str)
        match_day = (match_date - self.epoch).days
        
        # Sanitize inputs
        t1_p1 = _sanitize_name(t1_p1_name)
        t1_p2 = _sanitize_name(t1_p2_name)
        t2_p1 = _sanitize_name(t2_p1_name)
        t2_p2 = _sanitize_name(t2_p2_name)
        
        t1_players = [p for p in [t1_p1, t1_p2] if p]
        t2_players = [p for p in [t2_p1, t2_p2] if p]
        t1_key = frozenset(t1_players)
        t2_key = frozenset(t2_players)
        is_doubles = discipline in DOUBLES_DISCIPLINES

        # Feature dict - Must match training exactly
        feat = {}
        
        # --- ELO ---
        t1_elos = [self.tracker.get_elo(discipline, p) for p in t1_players]
        t2_elos = [self.tracker.get_elo(discipline, p) for p in t2_players]
        feat["elo_team_one"] = np.mean(t1_elos) if t1_elos else 1500.0
        feat["elo_team_two"] = np.mean(t2_elos) if t2_elos else 1500.0
        feat["elo_diff"] = feat["elo_team_one"] - feat["elo_team_two"]
        feat["elo_spread_team_one"] = abs(t1_elos[0] - t1_elos[1]) if is_doubles and len(t1_elos)==2 else 0.0
        feat["elo_spread_team_two"] = abs(t2_elos[0] - t2_elos[1]) if is_doubles and len(t2_elos)==2 else 0.0

        # --- Win Rates ---
        feat["win_rate_team_one"] = np.mean([self.tracker.get_win_rate(discipline, p) for p in t1_players])
        feat["win_rate_team_two"] = np.mean([self.tracker.get_win_rate(discipline, p) for p in t2_players])
        feat["win_rate_diff"] = feat["win_rate_team_one"] - feat["win_rate_team_two"]

        feat["h2h_win_rate"] = self.tracker.get_h2h_win_rate(discipline, t1_key, t2_key)
        
        # --- Rolling ---
        for w in ROLLING_WINDOWS:
            feat[f"win_rate_last{w}_team_one"] = np.mean([self.tracker.get_rolling_win_rate(discipline, p, w) for p in t1_players])
            feat[f"win_rate_last{w}_team_two"] = np.mean([self.tracker.get_rolling_win_rate(discipline, p, w) for p in t2_players])

        # --- Matches Played ---
        mp_t1 = sum(self.tracker.get_matches_played(discipline, p) for p in t1_players)
        mp_t2 = sum(self.tracker.get_matches_played(discipline, p) for p in t2_players)
        feat["matches_played_team_one"] = mp_t1
        feat["matches_played_team_two"] = mp_t2
        feat["log_matches_team_one"] = np.log1p(mp_t1)
        feat["log_matches_team_two"] = np.log1p(mp_t2)
        
        # --- Form ---
        feat["recent_form_team_one"] = np.mean([self.tracker.get_recent_form(discipline, p) for p in t1_players])
        feat["recent_form_team_two"] = np.mean([self.tracker.get_recent_form(discipline, p) for p in t2_players])
        
        # Defaults for complex tactical stats
        feat["avg_point_streak_team_one"] = np.mean([self.tracker.get_avg_streak(discipline, p) for p in t1_players])
        feat["avg_point_streak_team_two"] = np.mean([self.tracker.get_avg_streak(discipline, p) for p in t2_players])
        feat["max_streak_ever_team_one"] = max([self.tracker.get_max_streak_ever(discipline, p) for p in t1_players], default=0)
        feat["max_streak_ever_team_two"] = max([self.tracker.get_max_streak_ever(discipline, p) for p in t2_players], default=0)
        feat["streak_std_team_one"] = np.mean([self.tracker.get_streak_std(discipline, p) for p in t1_players])
        feat["streak_std_team_two"] = np.mean([self.tracker.get_streak_std(discipline, p) for p in t2_players])
        
        feat["pressure_wr_team_one"] = np.mean([self.tracker.get_pressure_win_rate(discipline, p) for p in t1_players])
        feat["pressure_wr_team_two"] = np.mean([self.tracker.get_pressure_win_rate(discipline, p) for p in t2_players])
        feat["pressure_opps_team_one"] = sum([self.tracker.get_pressure_opportunities(discipline, p) for p in t1_players])
        feat["pressure_opps_team_two"] = sum([self.tracker.get_pressure_opportunities(discipline, p) for p in t2_players])
        
        feat["close_game_pct_team_one"] = np.mean([self.tracker.get_close_game_pct(discipline, p) for p in t1_players])
        feat["close_game_pct_team_two"] = np.mean([self.tracker.get_close_game_pct(discipline, p) for p in t2_players])

        # --- Glicko-2 with Inflation ---
        def get_inflated_glicko(player_name):
            rating = self.glicko.get_rating(discipline, player_name)
            if rating.last_match_day is not None:
                 days_off = max(0, match_day - rating.last_match_day)
                 phi = rating.rd / 173.7178
                 phi_new = math.sqrt(phi**2 + rating.sigma**2 * days_off)
                 rd_inflated = min(phi_new * 173.7178, 350.0)
                 return rating.mu, rd_inflated, rating.sigma
            return rating.mu, rating.rd, rating.sigma

        t1_g = [get_inflated_glicko(p) for p in t1_players]
        t2_g = [get_inflated_glicko(p) for p in t2_players]
        
        feat["glicko_mu_team_one"] = np.mean([x[0] for x in t1_g]) if t1_g else 1500.0
        feat["glicko_mu_team_two"] = np.mean([x[0] for x in t2_g]) if t2_g else 1500.0
        feat["glicko_mu_diff"] = feat["glicko_mu_team_one"] - feat["glicko_mu_team_two"]
        
        feat["glicko_rd_team_one"] = np.mean([x[1] for x in t1_g]) if t1_g else 350.0
        feat["glicko_rd_team_two"] = np.mean([x[1] for x in t2_g]) if t2_g else 350.0
        feat["glicko_rd_diff"] = feat["glicko_rd_team_one"] - feat["glicko_rd_team_two"]
        
        feat["glicko_vol_team_one"] = np.mean([x[2] for x in t1_g]) if t1_g else 0.06
        feat["glicko_vol_team_two"] = np.mean([x[2] for x in t2_g]) if t2_g else 0.06

        # --- Context ---
        feat["round_encoded"] = ROUND_ORDINALS.get(round_name, 5)
        
        def check_home(nats, ctry):
            if not nats or not ctry: return 0.0
            valid_nats = [n.strip() for n in nats if n]
            return float(any(IOC_TO_COUNTRY.get(n, "") == ctry for n in valid_nats))
            
        feat["is_home_team_one"] = check_home([t1_p1_nat, t1_p2_nat], country)
        feat["is_home_team_two"] = check_home([t2_p1_nat, t2_p2_nat], country)
        
        # One-Hot
        req_cols = self.meta["feature_columns"]
        
        # Initialize 0s
        for col in req_cols:
            if col not in feat:
                feat[col] = 0.0

        if f"disc_{discipline}" in req_cols:
            feat[f"disc_{discipline}"] = 1.0
        for col in req_cols:
            if col.startswith("tourney_") and col.replace("tourney_", "") in tournament:
                feat[col] = 1.0
                
        # Predict
        X = pd.DataFrame([feat])[req_cols]
        # XGBClassifier.predict_proba returns (n_samples, n_classes) -> [[p_loss, p_win]]
        xgb_prob = self.xgb_model.predict_proba(X)[0][1]
        
        return {
            "t1_players": t1_players,
            "t2_players": t2_players,
            "prob": xgb_prob,
            "X": X, 
            "details": {
                "elo_t1": feat["elo_team_one"], "elo_t2": feat["elo_team_two"],
                "glicko_t1": feat["glicko_mu_team_one"], "glicko_t2": feat["glicko_mu_team_two"],
                "rd_t1": feat["glicko_rd_team_one"], "rd_t2": feat["glicko_rd_team_two"],
                "home_t1": feat["is_home_team_one"], "home_t2": feat["is_home_team_two"]
            }
        }
        
def calc_home_adv(h1, h2):
    return h1 - h2
