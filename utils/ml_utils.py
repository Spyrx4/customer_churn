import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

TENURE_BINS = [0, 6, 12, 24, 60, np.inf]
TENURE_LABELS = list(range(len(TENURE_BINS) - 1))

@st.cache_resource
def load_artifacts():
    # Gunakan absolute path atau relative dari root project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocess_path = os.path.join(base_dir, "artifacts", "preprocessing_fe", "preprocessing_artifacts.joblib")
    model_path = os.path.join(base_dir, "artifacts", "models", "xgb_ros.pkl")
    
    preprocess = joblib.load(preprocess_path)
    model = joblib.load(model_path)
    return preprocess, model

def predict_churn(raw_df: pd.DataFrame):
    """
    Melakukan pra-pemrosesan data mentah dan mengembalikan prediksi churn.
    Returns:
        predictions (array of int): 1 (Churn), 0 (Stay)
        probabilities (array of float): Probabilitas class 1
    """
    preprocess, model = load_artifacts()
    df = raw_df.copy()

    # Tenure binning
    df["Tenure_bucket"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=False)
    
    # Log transform
    for col in preprocess["log_cols"]:
        if col in df:
            df[col] = np.log1p(df[col])

    # Scaling
    df[preprocess["cont_cols"]] = preprocess["scaler"].transform(df[preprocess["cont_cols"]])
    
    # One-Hot Encoding
    X = preprocess["ohe_preprocess"].transform(df)

    # Inferensi
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    
    return predictions, probabilities
