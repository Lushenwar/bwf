# 🏸 Shuttle-X — Badminton Match Outcome Prediction Engine

## 1. Dataset Summary

### Files & Row Counts
| File | Discipline | Rows | Schema Type |
|------|-----------|------|-------------|
| `ms.csv` | Men's Singles (MS) | 3,761 | Singles |
| `ws.csv` | Women's Singles (WS) | 2,975 | Singles |
| `md.csv` | Men's Doubles (MD) | 2,804 | Doubles |
| `wd.csv` | Women's Doubles (WD) | 2,522 | Doubles |
| `xd.csv` | Mixed Doubles (XD) | 2,856 | Doubles |
| **Total** | | **14,918** | |

- **Date range**: Jan 2018 → Oct 2020 (format: `DD-MM-YYYY`)
- **Target column**: `winner` — values `1` (team_one wins), `2` (team_two wins), `0` (walkover/retirement — all `retired=True`)

### Schema Differences
- **Singles** (MS, WS): player columns are `team_one_players`, `team_two_players` (single name per field), nationality columns `team_one_nationalities`, `team_two_nationalities`.
- **Doubles** (MD, WD, XD): player columns are `team_one_player_one`, `team_one_player_two`, `team_two_player_one`, `team_two_player_two`, with separate nationality columns for each player.

### Common Columns (all 5 files)
| Column | Description | Leakage Risk |
|--------|-------------|--------------|
| `tournament` | Tournament name | ✅ Safe (pre-match) |
| `city`, `country` | Venue location | ✅ Safe |
| `date` | Match date (DD-MM-YYYY) | ✅ Safe (used for temporal sort) |
| `tournament_type` | e.g. Super 300/500/750/1000 | ✅ Safe |
| `discipline` | MS/WS/MD/WD/XD | ✅ Safe |
| `round` | Round of 32, QF, SF, Final, etc. | ✅ Safe |
| `winner` | **TARGET** — 0, 1, or 2 | 🎯 Target variable |
| `nb_sets` | Number of games played (1-3) | ⛔ LEAKAGE — post-match |
| `retired` | Whether a player retired | ⛔ LEAKAGE — post-match |
| `game_1_score` … `game_3_score` | Final game scores | ⛔ LEAKAGE — post-match |
| `team_*_total_points` | Total points scored | ⛔ LEAKAGE — post-match |
| `team_*_most_consecutive_points*` | Max point streak per game | ⛔ LEAKAGE — post-match |
| `team_*_game_points*` | Points list per game | ⛔ LEAKAGE — post-match |
| `game_*_scores` | Point-by-point score sequence | ⛔ LEAKAGE — post-match |

### Key Observations
1. **`winner = 0` rows** (~194 total) are ALL retirements (`retired=True`). We will **drop these** — they are not predictable outcomes.
2. **Game 3 columns** are null for ~65% of rows (matches that ended in 2 games). Game 2 columns are null for ~0.5% (walkovers after game 1 only).
3. **All score/point columns are POST-MATCH stats** → they CANNOT be used as input features. They will be used ONLY to derive historical (pre-match) features from PAST matches.
4. The `game_*_scores` column contains a **stringified Python list** of score progressions like `['0-0', '1-0', '2-0', ...]`. We will parse these from PAST matches to derive "point streak persistence" features.

---

## 2. Architecture Decision: One Global Model ✅

### Decision: **Single Global Model with discipline as a feature** (not five separate models)

### Rationale
| Factor | Global Model ✅ | Five Separate Models ❌ |
|--------|----------------|----------------------|
| Data volume | Uses all 14,918 rows | Smallest split has only 2,522 rows (WD) — XGBoost may underfit |
| Shared patterns | Tournament-level patterns (home advantage, round difficulty) transfer across disciplines | Siloed — no cross-discipline learning |
| Simplicity | One pipeline, one model, one deployment artifact | Five of everything — 5× maintenance burden |
| Discipline specificity | One-hot encoding of `discipline` lets the model learn discipline-specific splits internally | Naturally discipline-specific |
| Scalability | Adding a new discipline = adding a column value | Adding a new discipline = building a new pipeline |

> **One-Hot Encoding Explained (for a 16-year-old):**
> Imagine you have five categories: MS, WS, MD, WD, XD. A computer doesn't understand words, so we convert each category into a set of yes/no flags. For a Men's Singles match, the row gets: `MS=1, WS=0, MD=0, WD=0, XD=0`. For Mixed Doubles: `MS=0, WS=0, MD=0, WD=0, XD=1`. This way, the model can learn that "when XD=1, the patterns are different from when MS=1" — each discipline gets its own "lane" of logic inside the model's decision trees.

### ELO Tracking Strategy ✅ VERIFIED
- **Discipline-Isolated ELO**: A player's ELO rating is tracked **separately per discipline**. Kento Momota's MS ELO is completely independent from any doubles ELO he might have. Each discipline is a separate "bank account."
- **For doubles**: Each individual player gets their own ELO within that discipline. Team features are:
  - **Mean Team ELO**: `(player_one_elo + player_two_elo) / 2` — overall team strength.
  - **Partner ELO Spread**: `abs(player_one_elo - player_two_elo)` — detects if one player is "carrying" the team.
- **Cold-start default**: New players start at **1500 ELO** and 50% win rate. We keep them in training to avoid selection bias.

---

## 3. Feature Engineering Plan

### 3.1 Features We Will Build (all derived from PAST matches only)

| # | Feature | Description | Source |
|---|---------|-------------|--------|
| 1 | `elo_team_one` | Pre-match ELO for team one (avg of player ELOs for doubles) | Chronological ELO tracker |
| 2 | `elo_team_two` | Pre-match ELO for team two | Chronological ELO tracker |
| 3 | `elo_diff` | `elo_team_one - elo_team_two` | Derived |
| 4 | `elo_spread_team_one` | Partner ELO gap in doubles (0 for singles) — detects "carry" | Derived |
| 5 | `elo_spread_team_two` | Partner ELO gap in doubles (0 for singles) | Derived |
| 6 | `win_rate_team_one` | Discipline-specific historical win rate for team one | Rolling history |
| 7 | `win_rate_team_two` | Discipline-specific historical win rate for team two | Rolling history |
| 8 | `h2h_win_rate` | Head-to-head win rate (team one vs team two in this discipline) | H2H history |
| 9 | `matches_played_team_one` | Total matches played in this discipline (experience proxy) | Count |
| 10 | `matches_played_team_two` | Total matches played in this discipline | Count |
| 11 | `recent_form_team_one` | Win rate in last 10 matches (discipline-specific) | Sliding window |
| 12 | `recent_form_team_two` | Win rate in last 10 matches | Sliding window |
| 13 | `avg_point_streak_team_one` | Average max consecutive points across past games (from `game_*_scores`) | Parsed score lists |
| 14 | `avg_point_streak_team_two` | Average max consecutive points across past games | Parsed score lists |
| 15 | `tournament_type_encoded` | One-hot encoded tournament tier | Direct column |
| 16 | `round_encoded` | Ordinal encoding of round (higher = later stage) | Direct column |
| 17 | `discipline_encoded` | One-hot encoded discipline (MS/WS/MD/WD/XD) | Direct column |

### 3.2 Data Leakage Firewall 🔒

**Rule**: For match at index `N` (sorted chronologically), features are computed using ONLY matches `0 .. N-1`.

- ELO ratings are updated **after** each match — the pre-match ELO is the one that existed before the match happened.
- Win rates, form, and streaks are computed from the **historical window** up to (but not including) the current match.
- No `nb_sets`, `retired`, score, or point columns from the **current match** are ever used as features.

---

## 4. Training Strategy

### Pipeline
1. **Load & concatenate** all 5 CSVs into one DataFrame with a `discipline` column.
2. **Filter out** `winner = 0` (retirements) — these are unpredictable "black swan" events.
3. **Parse dates** (DD-MM-YYYY → datetime) and **sort chronologically**.
4. **Normalize player names** across singles/doubles schemas into a unified format.
5. **Run the feature engineering loop** chronologically (for each match, compute features from history, then update ELO/stats). Cold-start players get 1500 ELO / 50% win rate defaults.
6. **Train/Validation split**: Temporal split — train on first 80% of matches (by date), validate on the last 20%. **NOT** random split (to preserve temporal integrity).
8. **Train XGBoost** classifier (binary: team_one wins vs team_two wins, i.e., `winner` remapped to `1` → 1, `2` → 0).
9. **Evaluate** on validation set — target ≥ 70% accuracy, stretch goal 75%.
10. **Iterate** on feature engineering if < 70%.

### XGBoost Hyperparameters (Starting Point)
```python
{
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "seed": 42,
}
```

---

## 5. Project Structure

```
bwf-1/
├── data/                  # Raw CSV files (untouched)
│   ├── ms.csv
│   ├── ws.csv
│   ├── md.csv
│   ├── wd.csv
│   └── xd.csv
├── src/
│   ├── __init__.py
│   ├── config.py          # Constants, paths, hyperparameters
│   ├── data_loader.py     # Load, concatenate, clean, parse dates
│   ├── feature_engine.py  # ELO tracker, win rates, streaks, form
│   ├── train.py           # Training pipeline (load → features → split → train → evaluate)
│   └── predict.py         # Inference module (load model, predict new match)
├── models/                # Saved model artifacts (.json)
├── tasks/
│   └── todo.md            # This file
└── requirements.txt
```

---

## 6. TODO Checklist

### Phase 1: Setup & Data Loading
- [x] Create project structure (`src/`, `models/`, `requirements.txt`)
- [x] Write `requirements.txt` (pandas, xgboost, scikit-learn, numpy)
- [x] Write `src/config.py` — paths, hyperparameters, constants
- [x] Write `src/data_loader.py` — load all 5 CSVs, unify schemas (normalize singles vs doubles player columns), filter retirements, parse dates, sort by date

### Phase 2: Feature Engineering
- [x] Write `src/feature_engine.py` — ELO tracker class (discipline-isolated, per-player)
- [x] Implement historical win rate calculator (discipline-specific, per-player/team)
- [x] Implement head-to-head win rate feature
- [x] Implement recent form (last 10 matches sliding window)
- [x] Implement point streak persistence feature (parse `game_*_scores` from past matches)
- [x] Implement match experience counter
- [x] Wire all features into a single `build_features(df)` function that iterates chronologically

> ⚠️ **Data note**: `team_two_player_two` is always identical to `team_two_player_one` in all doubles CSVs (dataset limitation). This means `elo_spread_team_two` will always be 0. The model still works — team_two is effectively represented by a single player identity.

### Phase 3: Training Pipeline
- [x] Write `src/train.py` — temporal train/val split (80/20 by date)
- [x] Train XGBoost model and evaluate accuracy, log-loss, and feature importances
- [x] Save trained model to `models/` directory
- [ ] If accuracy < 70%: iterate on features (add/tune/remove)

> ✅ **Initial result**: 70.9% accuracy, 0.566 log-loss. Above 70% threshold. 
> Top feature: `elo_diff` (10.7%). Pushing for 75% stretch goal.

### Phase 4: Prediction Module
- [x] Write `src/predict.py` — load saved model, accept player names + discipline + tournament info, output win probability

### Phase 5: Audit & Hardening
- [x] **Temporal Integrity Audit**: DataFrame sorted by date ✅ (`is_monotonic_increasing=True`, range: 2018-01-09 → 2020-10-31)
- [x] **Discipline Isolation Audit**: ELO keys are `(discipline, player_name)` tuples ✅ (updating MS ELO leaves MD ELO untouched)
- [x] **Leakage Check**: Zero post-match columns in feature matrix ✅ (26 feature columns, all clean)
- [x] **Input Sanitization**: Empty names rejected, invalid disciplines rejected, lowercase normalized ✅
- [x] **Cold-Start Handling**: New players get ELO=1500, WinRate=0.5, Form=0.5 ✅

> 🔒 Full audit script: `audit.py` — run with `python audit.py`

### Phase 6: Documentation & Review
- [x] Add inline docstrings and type hints to all modules
- [x] Write review section in this file with summary of changes
- [x] Final accuracy report with confusion matrix and feature importance chart

---

## 7. Verified Decisions ✅

| # | Question | Decision | Rationale |
|---|----------|----------|----------|
| 1 | **Retirement handling** | **Drop all `winner=0`** (194 rows) | Retirements are "black swan" events — unpredictable noise that confuses XGBoost |
| 2 | **Cold-start threshold** | **Keep with 1500 ELO default** | Dropping cold-start players creates selection bias — upsets often come from "dark horse" unknowns |
| 3 | **Doubles ELO tracking** | **Individual ELO → team Mean + Spread** | Per-pair tracking is sparse; individual tracking + aggregation captures partner chemistry gaps |
| 4 | **Model output** | **Binary classification (team_one wins = 1, team_two wins = 0)** | Standard sports modeling format |

---

## 8. Review Section

### 8.1 Final Accuracy Report

| Metric | Value |
|--------|-------|
| **Accuracy** | **70.9%** |
| Log Loss | 0.566 |
| Training matches | 11,779 |
| Validation matches | 2,945 |
| Training date range | 2018-01-09 → 2020-03-15 |
| Validation date range | 2020-03-15 → 2020-10-31 |

### 8.2 Confusion Matrix

|  | Predicted: Team Two | Predicted: Team One |
|--|---------------------|---------------------|
| **Actual: Team Two** | 1,036 (TN) | 401 (FP) |
| **Actual: Team One** | 455 (FN) | 1,053 (TP) |

### 8.3 Top 10 Feature Importances

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `elo_diff` | 0.1070 |
| 2 | `matches_played_team_two` | 0.0483 |
| 3 | `matches_played_team_one` | 0.0456 |
| 4 | `tourney_Super_1000` | 0.0439 |
| 5 | `elo_team_one` | 0.0416 |
| 6 | `avg_point_streak_team_one` | 0.0410 |
| 7 | `avg_point_streak_team_two` | 0.0401 |
| 8 | `elo_team_two` | 0.0388 |
| 9 | `win_rate_team_one` | 0.0378 |
| 10 | `disc_WS` | 0.0377 |

### 8.4 Summary of Changes

| File | Purpose |
|------|---------|
| `requirements.txt` | Project dependencies (pandas, numpy, xgboost, scikit-learn) |
| `src/__init__.py` | Package init |
| `src/config.py` | Central config — paths, ELO params, hyperparameters, leakage blacklist |
| `src/data_loader.py` | Loads 5 CSVs, normalizes singles/doubles schemas, drops retirements, parses dates, sorts chronologically |
| `src/feature_engine.py` | Chronological feature builder — ELO tracker (discipline-isolated), win rates, H2H, recent form, point streak persistence, one-hot encoding |
| `src/train.py` | Training pipeline — data → features → leakage audit → temporal split → XGBoost → evaluation → save |
| `src/predict.py` | Inference module — loads model, validates inputs, returns win probabilities |
| `audit.py` | Automated audit — verifies temporal integrity, discipline isolation, leakage, sanitization, cold-start |
| `models/shuttle_x_model.json` | Saved XGBoost model |
| `models/shuttle_x_metadata.json` | Model metadata (accuracy, feature importances, params) |

### 8.5 Data Discovery: team_two_player_two Bug

During development we discovered that in ALL three doubles CSVs (`md.csv`, `wd.csv`, `xd.csv`), the `team_two_player_two` column always contains a copy of `team_two_player_one`. This is a **dataset limitation** — the second player on team two is never recorded. This means:
- `elo_spread_team_two` is always 0 (feature importance = 0.0)
- Team two in doubles is effectively represented by a single player identity
- Despite this limitation, the model achieves 70.9% accuracy

### 8.6 How One-Hot Encoding Works (for a 16-year-old)

Imagine you have five types of badminton: MS, WS, MD, WD, XD. A computer can't read words — it only understands numbers. So we create five new columns, one for each type, and fill them with 1 (yes) or 0 (no).

**Example:** A Men's Singles match gets coded as:

| MS | WS | MD | WD | XD |
|----|----|----|----|----|
| 1  | 0  | 0  | 0  | 0  |

A Mixed Doubles match gets coded as:

| MS | WS | MD | WD | XD |
|----|----|----|----|----|
| 0  | 0  | 0  | 0  | 1  |

This way, the XGBoost model can learn different "rules" for each discipline. If it notices that in Women's Singles (`WS=1`), certain players tend to win more often in the finals, it can create a decision tree split that says "IF WS=1 AND round=Final, THEN weight ELO difference more heavily." Each discipline gets its own "lane" in the model's decision-making process.|
