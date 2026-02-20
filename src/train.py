"""
train.py — Training pipeline for Shuttle-X.

What this script does (explained simply):
=========================================
1. Loads and cleans all the data (via data_loader).
2. Builds historical features for every match (via feature_engine).
3. Splits the data by TIME — the first 80% of matches become the
   "training set" and the last 20% become the "validation set."
   We do NOT shuffle randomly because that would let the model
   "peek into the future" (a form of data leakage).
4. Trains an XGBoost classifier to predict which team wins.
5. Evaluates accuracy and log-loss on the held-out validation set.
6. Saves the trained model to disk.

Usage:
    python -m src.train
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from xgboost import XGBClassifier

from src.config import (
    LEAKAGE_COLUMNS,
    MODELS_DIR,
    TRAIN_RATIO,
    XGB_PARAMS,
)
from src.data_loader import load_and_clean
from src.feature_engine import build_features, get_feature_columns

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Leakage audit — hard check before training
# ═══════════════════════════════════════════════════════════════════════════

def _audit_leakage(feature_cols: list[str]) -> None:
    """Verify that no post-match columns leaked into the feature set.

    Raises ValueError if a leakage column is found.
    """
    leaked = set(feature_cols) & LEAKAGE_COLUMNS
    if leaked:
        raise ValueError(
            f"🚨 DATA LEAKAGE DETECTED! The following post-match columns "
            f"are in the feature set: {leaked}"
        )
    logger.info("✅ Leakage audit passed — no post-match columns in features.")


# ═══════════════════════════════════════════════════════════════════════════
# Temporal train/validation split
# ═══════════════════════════════════════════════════════════════════════════

def temporal_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data by time — first N% train, last (100-N)% validation.

    This preserves temporal integrity: the model only trains on matches
    that happened BEFORE the validation matches.
    """
    split_idx = int(len(df) * train_ratio)

    X_train = df.iloc[:split_idx][feature_cols]
    X_val = df.iloc[split_idx:][feature_cols]
    y_train = df.iloc[:split_idx][target_col]
    y_val = df.iloc[split_idx:][target_col]

    logger.info("Temporal split — Train: %d rows, Val: %d rows (ratio=%.2f)",
                len(X_train), len(X_val), train_ratio)
    logger.info("Train dates: %s → %s",
                df.iloc[:split_idx]["date"].min().date(),
                df.iloc[:split_idx]["date"].max().date())
    logger.info("Val dates:   %s → %s",
                df.iloc[split_idx:]["date"].min().date(),
                df.iloc[split_idx:]["date"].max().date())

    return X_train, X_val, y_train, y_val


# ═══════════════════════════════════════════════════════════════════════════
# Train + evaluate
# ═══════════════════════════════════════════════════════════════════════════

def train_and_evaluate() -> dict:
    """Full training pipeline — load → features → split → train → evaluate.

    Returns a dict with accuracy, log_loss, feature_importances, etc.
    """
    # ── 1. Load & clean ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SHUTTLE-X TRAINING PIPELINE")
    logger.info("=" * 60)

    df = load_and_clean()

    # ── 2. Build features ──────────────────────────────────────────────
    featured_df = build_features(df)

    # ── 3. Get feature columns and audit ───────────────────────────────
    feature_cols = get_feature_columns(featured_df)
    _audit_leakage(feature_cols)

    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # ── 4. Temporal split ──────────────────────────────────────────────
    X_train, X_val, y_train, y_val = temporal_split(featured_df, feature_cols)

    # ── 5. Train XGBoost ───────────────────────────────────────────────
    logger.info("Training XGBoost with params: %s", XGB_PARAMS)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── 6. Evaluate ────────────────────────────────────────────────────
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    ll = log_loss(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=["Team Two Wins", "Team One Wins"])

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("Accuracy:  %.4f (%.1f%%)", acc, acc * 100)
    logger.info("Log Loss:  %.4f", ll)
    logger.info("\nConfusion Matrix:\n%s", cm)
    logger.info("\nClassification Report:\n%s", report)

    # Feature importances
    importances = dict(zip(feature_cols, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    logger.info("\nTop 10 Feature Importances:")
    for fname, fimp in sorted_imp[:10]:
        logger.info("  %-35s  %.4f", fname, fimp)

    # ── 7. Save model ─────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "shuttle_x_model.json"
    model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    # Save metadata alongside the model
    metadata = {
        "accuracy": float(acc),
        "log_loss": float(ll),
        "feature_columns": feature_cols,
        "feature_importances": {k: float(v) for k, v in sorted_imp},
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "xgb_params": {k: v for k, v in XGB_PARAMS.items() if k != "seed"},
        "confusion_matrix": cm.tolist(),
    }
    meta_path = MODELS_DIR / "shuttle_x_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    return metadata


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    results = train_and_evaluate()

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  Accuracy:  {results['accuracy']:.1%}")
    print(f"  Log Loss:  {results['log_loss']:.4f}")
    print(f"  Train:     {results['train_size']} matches")
    print(f"  Val:       {results['val_size']} matches")

    if results["accuracy"] < 0.70:
        print("\n⚠️  Accuracy below 70% — feature iteration needed!")
    elif results["accuracy"] >= 0.75:
        print("\n🎯  Target 75% reached! 🚀")
    else:
        print("\n✅  Above 70% — solid baseline. Stretch goal is 75%.")
