"""
config.py — Central configuration for Shuttle-X.

Think of this file as the "settings panel" for the entire project.
Every magic number, file path, and tunable knob lives here so we
never have to hunt through code to change something.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Raw CSV filenames — one per discipline
CSV_FILES: dict[str, str] = {
    "MS": "ms.csv",
    "WS": "ws.csv",
    "MD": "md.csv",
    "WD": "wd.csv",
    "XD": "xd.csv",
}

# Disciplines classified by type (affects how we read player columns)
SINGLES_DISCIPLINES: set[str] = {"MS", "WS"}
DOUBLES_DISCIPLINES: set[str] = {"MD", "WD", "XD"}

# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
DATE_FORMAT = "%d-%m-%Y"  # DD-MM-YYYY as found in the CSVs

# ---------------------------------------------------------------------------
# ELO system
# ---------------------------------------------------------------------------
ELO_DEFAULT = 1500        # Starting ELO for unseen players
ELO_K_FACTOR = 32         # How fast ELO reacts to results (standard chess K)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
RECENT_FORM_WINDOW = 10   # Number of recent matches for form calculation
DEFAULT_WIN_RATE = 0.5    # Default win rate for cold-start players

# ---------------------------------------------------------------------------
# Round ordinal encoding (higher = deeper into tournament = harder opponent)
# ---------------------------------------------------------------------------
ROUND_ORDINALS: dict[str, int] = {
    "Qualification round of 32": 1,
    "Qualification round of 16": 2,
    "Qualification quarter final": 3,
    "Round of 64": 4,
    "Round of 32": 5,
    "Round of 16": 6,
    "Round 1": 5,   # Group-stage rounds mapped to similar depth
    "Round 2": 6,
    "Round 3": 7,
    "Quarter final": 8,
    "Semi final": 9,
    "Final": 10,
}

# ---------------------------------------------------------------------------
# XGBoost hyperparameters
# ---------------------------------------------------------------------------
XGB_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "seed": 42,
    "verbosity": 1,
}

# ---------------------------------------------------------------------------
# Train / Validation split
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.80  # First 80 % of matches (by date) are training set

# ---------------------------------------------------------------------------
# Columns flagged as POST-MATCH (leakage risks) — never use as features
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS: set[str] = {
    "nb_sets",
    "retired",
    "game_1_score",
    "game_2_score",
    "game_3_score",
    "team_one_total_points",
    "team_two_total_points",
    "team_one_most_consecutive_points",
    "team_two_most_consecutive_points",
    "team_one_game_points",
    "team_two_game_points",
    "team_one_most_consecutive_points_game_1",
    "team_two_most_consecutive_points_game_1",
    "team_one_game_points_game_1",
    "team_two_game_points_game_1",
    "game_1_scores",
    "team_one_most_consecutive_points_game_2",
    "team_two_most_consecutive_points_game_2",
    "team_one_game_points_game_2",
    "team_two_game_points_game_2",
    "game_2_scores",
    "team_one_most_consecutive_points_game_3",
    "team_two_most_consecutive_points_game_3",
    "team_one_game_points_game_3",
    "team_two_game_points_game_3",
    "game_3_scores",
}

# Allowed discipline values (for input sanitization)
VALID_DISCIPLINES: set[str] = {"MS", "WS", "MD", "WD", "XD"}
