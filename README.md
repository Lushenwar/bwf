## Introducing BWF 
An end-to-end Machine Learning pipeline that predicts BWF World Tour match outcomes with 75% accuracy (targeting 80%+). This project utilizes historical point-by-point data, individual Glicko-2 skill tracking, and gradient-boosted ensembles to forecast the next generation of badminton legends.

I used the dataset of badminton matches across 88 different BWF World Tour competition from 2018 to 2021 that I got from kaggle dataset down below.

Data Source : https://www.kaggle.com/sanderp/badminton-bwf-world-tour

### 🚀 Key Features
Glicko-2 Rating System: Advanced skill tracking that accounts for rating uncertainty (RD) and player volatility.

Momentum Analysis: Features extracted from game_i_scores to identify "Clutch" performers (18-18 tie specialists).

Interactive Dashboard: A Streamlit UI for head-to-head predictions and "Force Plots" explaining the model's logic.

Tournament Simulator: Recursive bracket simulation to predict entire tournament draws (e.g., All England Open).

Home Advantage Logic: Quantifies the physiological and psychological boost of playing in one's home country.

### 🛠️ Tech Stack
Model: LightGBM + XG Boost Ensemble

Optimization: Optuna (Bayesian Hyperparameter Tuning)

Explainability: SHAP (Shapley Additive Explanations)

App Framework: Streamlit

Data Manipulation: Pandas, NumPy

### 📂 Project Structure
```
├── data/                # Raw and processed BWF CSVs
├── models/              # Trained .pkl models and scalers
├── src/
│   ├── app.py           # Streamlit UI code
│   ├── train.py         # ML training pipeline
│   ├── predict.py       # Inference & Tournament simulation
│   └── utils/           # Glicko-2 and cleaning helpers
├── tasks/               # todo.md and project tracking
└── requirements.txt     # Production dependencies
```
### 🚦 Getting Started
1. Prerequisites:
   
  Ensure you have Python 3.9+ installed.

2. Installation
   
  Bash
```
git clone https://github.com/Lushenwar/bwf.git
cd bwf
pip install -r requirements.txt
```
3. Run the Dashboard
   
  Bash
```
streamlit run src/app.py
```

Note: Included sample datasets in bwf/data
### 📊 Results & Performance
Current Accuracy: 75.2% on 2021 BWF Validation Set.

Key Insight: Home Advantage and "Late-Set Points Won" are currently the highest-weighted features according to SHAP analysis.

### 🔒 Security & Best Practices
No Data Leakage: All features are "lagged" to ensure the model never sees the future.

Sanitized Inputs: All user-provided tournament draws are validated before inference.

Environment Variables: Sensitive configurations are managed via .env (not included in repo).
