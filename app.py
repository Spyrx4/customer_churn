import streamlit as st
import pandas as pd
import joblib
from views import batch_predict
from views.ui_analytics import render_analytics
from views.ui_predict import render_prediction
from views.ui_agent import render_agent

# Page configuration
st.set_page_config(
    page_title="MixxComm Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "sim_data" not in st.session_state:
    st.session_state["sim_data"] = {}


# Custom styling
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%); }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px; padding: 20px 24px; backdrop-filter: blur(12px);
    }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e1b4b 0%, #1a1a3e 100%); }
    .dashboard-subheader { color: #a5b4fc; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; }
    .pred-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px; padding: 28px 32px; text-align: center; backdrop-filter: blur(12px);
    }
    .risk-high { border-left: 4px solid #f43f5e; }
    .risk-medium { border-left: 4px solid #fb923c; }
    .risk-low { border-left: 4px solid #34d399; }
    .agent-greeting {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 16px; padding: 24px 28px;
    }
</style>
""", unsafe_allow_html=True)

# Data & Model loading
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    return df.drop(columns=["id"], errors="ignore")

df = load_data()

# Sidebar filters
st.sidebar.markdown("## Configuration")
user_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Paste your OpenAI API Key here for the AI Consultant demo.")
if user_api_key:
    import os
    os.environ["OPENAI_API_KEY"] = user_api_key
    st.session_state["OPENAI_API_KEY"] = user_api_key

st.sidebar.markdown("---")
st.sidebar.markdown("## Filters")
gender_filter = st.sidebar.multiselect("Gender", options=df["gender"].unique(), default=df["gender"].unique())
contract_filter = st.sidebar.multiselect("Contract", options=df["Contract"].unique(), default=df["Contract"].unique())
internet_filter = st.sidebar.multiselect("Internet Service", options=df["InternetService"].unique(), default=df["InternetService"].unique())
senior_filter = st.sidebar.multiselect("Senior Citizen", options=[0, 1], default=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

mask = (df["gender"].isin(gender_filter) & df["Contract"].isin(contract_filter) & 
        df["InternetService"].isin(internet_filter) & df["SeniorCitizen"].isin(senior_filter))
filtered = df[mask].copy()
st.sidebar.markdown(f"Total: {len(filtered):,} / {len(df):,} customers")

# Main interface
st.markdown("# MixxComm Churn Analytics")
st.markdown('<p style="color:#94a3b8; margin-top:-10px;">Analitik & Prediksi Churn Pelanggan</p>', unsafe_allow_html=True)

tabs = st.tabs(["Global Analytics", "Individual Prediction & Simulation", "Batch Prediction", "AI Consultant (Rini)"])

with tabs[0]:
    churn_count = (filtered["Churn"] == "Yes").sum()
    churn_rate = churn_count / len(filtered) * 100 if len(filtered) > 0 else 0
    render_analytics(filtered, churn_rate)

with tabs[1]:
    render_prediction()

with tabs[2]:
    batch_predict.render()

with tabs[3]:
    render_agent()

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center; color:#64748b; font-size:13px;">MixxComm Dashboard | 594K Records | XGBoost</p>', unsafe_allow_html=True)
