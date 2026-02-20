"""Audit script for Shuttle-X — verifies all integrity guarantees."""
import logging
logging.basicConfig(level=logging.WARNING)

from src.data_loader import load_and_clean
from src.feature_engine import build_features, get_feature_columns, PlayerTracker
from src.config import LEAKAGE_COLUMNS, ELO_DEFAULT, DEFAULT_WIN_RATE
from src.predict import _sanitize_player_name, _validate_discipline

results = []

# ── AUDIT 1: TEMPORAL INTEGRITY ────────────────────────────────────────
df = load_and_clean()
ok = df["date"].is_monotonic_increasing
results.append(("Temporal Integrity", ok,
                 f"is_monotonic_increasing={ok}, "
                 f"range={df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}"))

# ── AUDIT 2: DISCIPLINE ISOLATION ──────────────────────────────────────
tracker = PlayerTracker()
# Simulate: update ELO for a player in MS, check that MD is unaffected
tracker.update_elo("MS", "Test Player", "Opponent")
ms_elo = tracker.get_elo("MS", "Test Player")
md_elo = tracker.get_elo("MD", "Test Player")
ok2 = ms_elo != ELO_DEFAULT and md_elo == ELO_DEFAULT
results.append(("Discipline Isolation", ok2,
                 f"MS ELO={ms_elo:.1f} (changed), MD ELO={md_elo:.1f} (untouched)"))

# ── AUDIT 3: LEAKAGE CHECK ────────────────────────────────────────────
featured = build_features(df)
feat_cols = get_feature_columns(featured)
leaked = set(feat_cols) & LEAKAGE_COLUMNS
ok3 = len(leaked) == 0
results.append(("Leakage Check", ok3,
                 f"Leaked columns: {leaked if leaked else 'NONE'}"))

# ── AUDIT 4: INPUT SANITIZATION ───────────────────────────────────────
try:
    _sanitize_player_name("")
    ok4a = False
except ValueError:
    ok4a = True

try:
    _validate_discipline("INVALID")
    ok4b = False
except ValueError:
    ok4b = True

try:
    result = _validate_discipline("ms")
    ok4c = result == "MS"
except ValueError:
    ok4c = False

ok4 = ok4a and ok4b and ok4c
results.append(("Input Sanitization", ok4,
                 f"Empty name rejected={ok4a}, Invalid disc rejected={ok4b}, "
                 f"Lowercase normalized={ok4c}"))

# ── AUDIT 5: COLD-START DEFAULTS ──────────────────────────────────────
tracker2 = PlayerTracker()
new_elo = tracker2.get_elo("MS", "Ghost Player")
new_wr = tracker2.get_win_rate("MS", "Ghost Player")
new_form = tracker2.get_recent_form("MS", "Ghost Player")
ok5 = (new_elo == ELO_DEFAULT and new_wr == DEFAULT_WIN_RATE
       and new_form == DEFAULT_WIN_RATE)
results.append(("Cold-Start Defaults", ok5,
                 f"ELO={new_elo}, WinRate={new_wr}, Form={new_form}"))

# ── PRINT RESULTS ─────────────────────────────────────────────────────
print("=" * 60)
print("SHUTTLE-X AUDIT RESULTS")
print("=" * 60)
all_pass = True
for name, passed, detail in results:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {name}")
    print(f"         {detail}")

print("=" * 60)
if all_pass:
    print("ALL AUDITS PASSED ✅")
else:
    print("SOME AUDITS FAILED ❌")
