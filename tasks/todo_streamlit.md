# Shuttle-X Live Dashboard (Streamlit)

> **Objective:** Build a production-ready interactive dashboard to predict match outcomes and visualize model logic.

## L1. Tech Stack & Setup
- [ ] **Dependencies**: Install `streamlit`, `streamlit-shap`, `plotly`.
- [ ] **Data Caching**: Use `@st.cache_resource` for loading the `BatchPredictor` (model + 15k match replay).
- [ ] **Feature Exposure**: Modify `BatchPredictor` to expose:
  - `get_known_players(discipline)`: Returns list of known players.
  - `predict(...)`: Return raw feature DataFrame `X` for SHAP usage.

## L2. UI Components (Side-by-Side)
- [ ] **Sidebar Controls**:
  - `st.selectbox` for Discipline (MS, WS, MD, WD, XD).
  - `st.date_input` for Match Date (affects Glicko RD).
  - `st.selectbox` for Tournament & Country (home advantage).
- [ ] **Player Selection**:
  - `col1, col2 = st.columns(2)`
  - Searchable dropdowns for Team 1 and Team 2.
  - Handle "Cold Start" (manual entry for new players).
- [ ] **Prediction Action**:
  - "Predict Outcome" button (primary action).
  - Spinner/Progress bar during inference.

## L3. Visualization (Plotly + SHAP)
- [ ] **Win Gauge**: `go.Indicator` (Plotly) showing Team 1's win probability.
  - Ranges: 0-25% (Red), 25-50% (Orange), 50-75% (Light Green), 75-100% (Green).
- [ ] **SHAP Explainability**:
  - `streamlit_shap.st_shap(shap.force_plot(...))`
  - Shows which features pushed the probability up/down.
- [ ] **Tale of the Tape**:
  - Comparative metrics table/chart:
    - ELO vs ELO
    - Glicko RD (Activity confidence)
    - Clutch Factor (Pressure Win Rate)
    - Recent Form (Last 5 win %)

## L4. Tournament Simulator Mode
- [ ] **Bracket Upload**:
  - Tab for "Tournament Simulator".
  - Upload CSV with columns: `Round, Player1, Nat1, Player2, Nat2`.
- [ ] **Recursive Simulation**:
  - Run prediction for R32 -> Winners move to R16 -> ... -> Final.
- [ ] **Results Table**:
  - Display full path to victory.

## L5. Security & Cleanup
- [ ] **Audit**: Ensure no secrets in code.
- [ ] **Performance**: Verify cache hits (load time < 2s after first run).
