"""
config.py — Central configuration for Shuttle-X.
"""

from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Raw CSV filenames
CSV_FILES: dict[str, str] = {
    "MS": "ms.csv",
    "WS": "ws.csv",
    "MD": "md.csv",
    "WD": "wd.csv",
    "XD": "xd.csv",
}

SINGLES_DISCIPLINES: set[str] = {"MS", "WS"}
DOUBLES_DISCIPLINES: set[str] = {"MD", "WD", "XD"}

# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
DATE_FORMAT = "%d-%m-%Y"

# ---------------------------------------------------------------------------
# ELO system
# ---------------------------------------------------------------------------
ELO_DEFAULT = 1500
ELO_K_FACTOR = 32

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
RECENT_FORM_WINDOW = 10
DEFAULT_WIN_RATE = 0.5
ROLLING_WINDOWS = [3, 5, 10]  # Rolling win rate windows (last N matches)

# ---------------------------------------------------------------------------
# Round ordinal encoding
# ---------------------------------------------------------------------------
ROUND_ORDINALS: dict[str, int] = {
    "Qualification round of 32": 1,
    "Qualification round of 16": 2,
    "Qualification quarter final": 3,
    "Round of 64": 4,
    "Round of 32": 5,
    "Round of 16": 6,
    "Round 1": 5,
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
TRAIN_RATIO = 0.80  # First 80% matches are training set
VALIDATION_CUTOFF = datetime(2021, 1, 1) # Keeping this just in case other scripts use it

# ---------------------------------------------------------------------------
# Leakage Columns
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS: set[str] = {
    "nb_sets", "retired",
    "game_1_score", "game_2_score", "game_3_score",
    "team_one_total_points", "team_two_total_points",
    "team_one_most_consecutive_points", "team_two_most_consecutive_points",
    "team_one_game_points", "team_two_game_points",
    "team_one_most_consecutive_points_game_1", "team_two_most_consecutive_points_game_1",
    "team_one_game_points_game_1", "team_two_game_points_game_1",
    "game_1_scores",
    "team_one_most_consecutive_points_game_2", "team_two_most_consecutive_points_game_2",
    "team_one_game_points_game_2", "team_two_game_points_game_2",
    "game_2_scores",
    "team_one_most_consecutive_points_game_3", "team_two_most_consecutive_points_game_3",
    "team_one_game_points_game_3", "team_two_game_points_game_3",
    "game_3_scores",
}

VALID_DISCIPLINES: set[str] = {"MS", "WS", "MD", "WD", "XD"}

# ---------------------------------------------------------------------------
# Elite Feature Maps
# ---------------------------------------------------------------------------

# Map IOC 3-letter codes to Full Tournament Country Names for Home Advantage
IOC_TO_COUNTRY: dict[str, str] = {
    "THA": "Thailand",
    "INA": "Indonesia",
    "IND": "India",
    "CHN": "China",
    "FRA": "France",
    "MAS": "Malaysia",
    "GER": "Germany",
    "KOR": "Korea",
    "JPN": "Japan",
    "ENG": "England",
    "DEN": "Denmark",
    "TPE": "Taipei", 
    "HKG": "Hong Kong",
    "SGP": "Singapore",
    "AUS": "Australia",
    "CAN": "Canada",
    "VIE": "Vietnam",
    "RUS": "Russia",
    "ESP": "Spain",
    "NZL": "New Zealand",
    "SUI": "Switzerland",
    "NED": "Netherlands",
    "USA": "U.S.A.",
    "MAC": "Macau",
}

# Normalize diverse city names
CITY_NORMALIZATION: dict[str, str] = {
    "PARIS": "Paris",
    "ORLEANS": "Orleans",
    "Saarbrucken": "Saarbrücken",
    "Odense V": "Odense",
    "Macau City": "Macau",
    "Taipei City": "Taipei",
    "Gwangju Metropolitan City": "Gwangju",
    "Ling Shui": "Lingshui",
    "Vladivostock": "Vladivostok",
    "Bangkok": "Bangkok",
    "Jakarta": "Jakarta",
}
