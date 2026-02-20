import sys
import os

# Robustly add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import xgboost as xgb
import shap
from streamlit_shap import st_shap

from src.batch_predictor import BatchPredictor
from src.config import IOC_TO_COUNTRY

# --- Page Config ---
st.set_page_config(
    page_title="Shuttle-X Live",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Styling (Zuckerberg Mode: Clean & Fast) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        height: 60px;
        font-size: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading (Cached) ---
@st.cache_resource
def load_predictor():
    predictor = BatchPredictor()
    predictor.load()
    return predictor

try:
    with st.spinner("Initializing Shuttle-X Neural Core..."):
        predictor = load_predictor()
except Exception as e:
    st.error(f"Failed to load model system: {e}")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.title("Match Setup")

discipline = st.sidebar.selectbox("Discipline", ["MS", "WS", "MD", "WD", "XD"])
match_date = st.sidebar.date_input("Match Date", datetime.today())

# Dynamic Player Lists
known_players = predictor.get_known_players(discipline)
known_players.insert(0, "") # Default empty option

# Tournament Context
country_options = sorted(list(set(IOC_TO_COUNTRY.values())))
country = st.sidebar.selectbox("Tournament Country", country_options, 
                             index=country_options.index("England") if "England" in country_options else 0)
tournament = st.sidebar.text_input("Tournament Name", "All England Open")

# --- Tabs ---
tab1, tab2 = st.tabs(["Head-to-Head", "Tournament Simulator"])

# ==============================================================================
# TAB 1: HEAD-TO-HEAD
# ==============================================================================
with tab1:
    st.title("Shuttle-X Live")
    st.markdown("### Real-time Badminton Match Prediction Engine")
    
    round_name = st.selectbox("Round", ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"])

    col1, col_vs, col2 = st.columns([5, 1, 5])

    with col1:
        st.subheader("Team One")
        t1_p1_name = st.selectbox("Player 1", known_players, key="t1_p1") 
        t1_p1_nat = st.text_input("Nat (IOC)", value="DEN" if "Axelsen" in str(t1_p1_name) else "", key="t1_n1").upper()
        
        t1_p2_name = None
        t1_p2_nat = None
        if discipline in ["MD", "WD", "XD"]:
            t1_p2_name = st.selectbox("Player 2", known_players, key="t1_p2")
            t1_p2_nat = st.text_input("Nat (IOC)", key="t1_n2").upper()

    with col_vs:
        st.markdown("<h1 style='text-align: center; padding-top: 100px;'>VS</h1>", unsafe_allow_html=True)

    with col2:
        st.subheader("Team Two")
        t2_p1_name = st.selectbox("Player 1", known_players, key="t2_p1") 
        t2_p1_nat = st.text_input("Nat (IOC)", value="MAS" if "Lee" in str(t2_p1_name) else "", key="t2_n1").upper()
        
        t2_p2_name = None
        t2_p2_nat = None
        if discipline in ["MD", "WD", "XD"]:
            t2_p2_name = st.selectbox("Player 2", known_players, key="t2_p2")
            t2_p2_nat = st.text_input("Nat (IOC)", key="t2_n2").upper()

    # --- Prediction & Visualization ---
    predict_btn = st.button("PREDICT OUTCOME", key="btn_h2h")

    if predict_btn:
        if not t1_p1_name or not t2_p1_name:
            st.warning("Please select at least Player 1 for both teams.")
        else:
            with st.spinner("Running Inference..."):
                try:
                    res = predictor.predict(
                        date_str=str(match_date),
                        discipline=discipline,
                        t1_p1_name=t1_p1_name, t1_p1_nat=t1_p1_nat, t1_p2_name=t1_p2_name, t1_p2_nat=t1_p2_nat,
                        t2_p1_name=t2_p1_name, t2_p1_nat=t2_p1_nat, t2_p2_name=t2_p2_nat, t2_p2_nat=t2_p2_nat,
                        tournament=tournament,
                        country=country,
                        round_name=round_name
                    )
                    
                    prob = res["prob"]
                    details = res["details"]
                    X = res["X"]

                    # 1. Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        title = {'text': f"Win Probability: {t1_p1_name}"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "black"},
                            'steps' : [
                                {'range': [0, 25], 'color': "#FF4B4B"},
                                {'range': [25, 50], 'color': "#FFA500"},
                                {'range': [50, 75], 'color': "#90EE90"},
                                {'range': [75, 100], 'color': "#008000"}
                            ],
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    # 2. Tale of the Tape
                    st.subheader("Tale of the Tape")
                    m1, m2, m3, m4 = st.columns(4)
                    
                    with m1:
                        diff = details['elo_t1'] - details['elo_t2']
                        st.metric("ELO Diff", f"{diff:.0f}", delta_color="normal")
                        st.caption(f"T1: {details['elo_t1']:.0f} | T2: {details['elo_t2']:.0f}")

                    with m2:
                        g_diff = details['glicko_t1'] - details['glicko_t2']
                        st.metric("Glicko Diff", f"{g_diff:.0f}", delta_color="normal")
                        st.caption(f"T1: {details['glicko_t1']:.0f} | T2: {details['glicko_t2']:.0f}")

                    with m3:
                        rd_diff = details['rd_t1'] - details['rd_t2']
                        st.metric("Uncertainty (RD) Diff", f"{rd_diff:.1f}", help="Positive means T1 is MORE uncertain")
                        st.caption(f"T1: {details['rd_t1']:.0f} | T2: {details['rd_t2']:.0f}")
                    
                    with m4:
                        home_adv = "Team 1" if details['home_t1'] else ("Team 2" if details['home_t2'] else "Neutral")
                        st.metric("Home Advantage", home_adv)

                    # 3. SHAP Explanation
                    st.subheader("Model Logic (Real-Time SHAP)")
                    try:
                        # Use TreeExplainer on the updated X
                        explainer = shap.TreeExplainer(predictor.xgb_model)
                        shap_values = explainer.shap_values(X)
                        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]), height=150)
                    except Exception as shap_err:
                        st.warning(f"SHAP visualization failed: {shap_err}")

                except Exception as e:
                    st.error(f"Prediction Error: {e}")

# ==============================================================================
# TAB 2: TOURNAMENT SIMULATOR
# ==============================================================================
with tab2:
    st.header("Tournament Simulator")
    st.markdown("Upload a CSV with columns: `Player1, Nat1, Player2, Nat2` (representing Round of 32 matchups).")
    
    uploaded_file = st.file_uploader("Upload Bracket CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            bracket_df = pd.read_csv(uploaded_file)
            st.dataframe(bracket_df.head())
            
            if st.button("Run Simulation", key="btn_sim"):
                results = []
                rounds = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
                
                # Convert DF to list of dicts
                current_round = []
                for _, row in bracket_df.iterrows():
                    current_round.append({
                        "p1": row['Player1'], "n1": row['Nat1'],
                        "p2": row['Player2'], "n2": row['Nat2']
                    })
                
                sim_log = st.empty()
                
                for r_name in rounds:
                    if not current_round:
                        break
                        
                    next_round_players = []
                    round_results = []
                    
                    sim_log.text(f"Simulating {r_name}...")
                    
                    for match in current_round:
                        res = predictor.predict(
                            date_str=str(match_date),
                            discipline=discipline,
                            t1_p1_name=match['p1'], t1_p1_nat=match['n1'], t1_p2_name=None, t1_p2_nat=None,
                            t2_p1_name=match['p2'], t2_p1_nat=match['n2'], t2_p2_name=None, t2_p2_nat=None,
                            tournament=tournament,
                            country=country,
                            round_name=r_name
                        )
                        
                        winner_name = res["t1_players"][0] if res["prob"] > 0.5 else res["t2_players"][0]
                        winner_nat = match['n1'] if res["prob"] > 0.5 else match['n2']
                        conf = res["prob"] if res["prob"] > 0.5 else 1 - res["prob"]
                        
                        round_results.append({
                            "Round": r_name,
                            "Match": f"{match['p1']} vs {match['p2']}",
                            "Winner": winner_name,
                            "Confidence": f"{conf:.1%}"
                        })
                        
                        next_round_players.append({"name": winner_name, "nat": winner_nat})
                    
                    results.extend(round_results)
                    
                    # Pair up for next round
                    current_round = []
                    for i in range(0, len(next_round_players), 2):
                        if i+1 < len(next_round_players):
                            current_round.append({
                                "p1": next_round_players[i]['name'], "n1": next_round_players[i]['nat'],
                                "p2": next_round_players[i+1]['name'], "n2": next_round_players[i+1]['nat']
                            })
                            
                st.success("Simulation Complete!")
                st.write("### Tournament Results")
                st.table(pd.DataFrame(results))
                
                if next_round_players:
                    st.balloons()
                    st.markdown(f"## Champion: {next_round_players[0]['name']}")

        except Exception as e:
            st.error(f"Error reading CSV or simulating: {e}")

# --- Footer ---
st.markdown("---")
