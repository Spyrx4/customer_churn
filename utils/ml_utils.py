import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import shap
import warnings

# Mengabaikan warning versi (scikit-learn & xgboost) saat me-load pickle dari versi yang sedikit berbeda
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

TENURE_BINS = [0, 6, 12, 24, 60, np.inf]
TENURE_LABELS = list(range(len(TENURE_BINS) - 1))

@st.cache_resource
def load_artifacts():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preprocess_path = os.path.join(base_dir, "artifacts", "preprocessing_fe", "preprocessing_artifacts.joblib")
        model_path = os.path.join(base_dir, "artifacts", "models", "xgb_ros.pkl")
        
        if not os.path.exists(preprocess_path) or not os.path.exists(model_path):
            st.error("Model artifacts not found. Please ensure the model is trained.")
            return None, None
            
        preprocess = joblib.load(preprocess_path)
        model = joblib.load(model_path)
        return preprocess, model
    except Exception as e:
        st.error(f"Critical error loading ML artifacts: {e}")
        return None, None

def predict_churn(raw_df: pd.DataFrame):
    """
    Melakukan pra-pemrosesan data mentah dan mengembalikan prediksi churn.
    Returns:
        predictions (array of int): 1 (Churn), 0 (Stay)
        probabilities (array of float): Probabilitas class 1
    """
    preprocess, model = load_artifacts()
    if preprocess is None or model is None:
        return [0], [0.0]
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

def explain_prediction(raw_df: pd.DataFrame):
    """
    Menghitung SHAP values untuk local explanation (1 data point).
    """
    preprocess, model = load_artifacts()
    if preprocess is None or model is None:
        return None
        
    df = raw_df.copy()

    df["Tenure_bucket"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=False)
    
    for col in preprocess["log_cols"]:
        if col in df:
            df[col] = np.log1p(df[col])

    df[preprocess["cont_cols"]] = preprocess["scaler"].transform(df[preprocess["cont_cols"]])
    
    X = preprocess["ohe_preprocess"].transform(df)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Ekstrak feature names dari transformer
        feature_names = preprocess["ohe_preprocess"].get_feature_names_out()
        
        # Ambil baris pertama (karena ini local explanation / 1 baris)
        # SHAP explainer XGBoost biasanya mereturn matriks (samples, features) 
        # Jika output list (multi-class), ambil index 1 untuk class 'Churn'
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]
            
        res_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": vals
        })
        
        return res_df
    except Exception:
        # Return None gracefully if SHAP fails due to XGBoost version incompatibility
        return None
