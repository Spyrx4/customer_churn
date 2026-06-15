import streamlit as st
import pandas as pd
import numpy as np
import joblib

def inject_simulator():
    st.markdown("---")
    st.markdown("### Churn Prevention Simulator")
    st.markdown("Ubah variabel di bawah untuk simulasi strategi bisnis.")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.write("Input Strategi")
        new_monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, key="sim_monthly")
        new_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="sim_contract")
        new_tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], index=1, key="sim_tech")
        new_tenure = st.slider("Tenure (bulan)", 1, 72, 12, key="sim_tenure")

    raw_data = pd.DataFrame([{
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': new_tenure,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': new_tech,
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': new_contract,
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': new_monthly,
        'TotalCharges': new_monthly * new_tenure 
    }])

    try:
        from utils.ml_utils import predict_churn
        df = raw_data.copy()
        
        # Inference
        predictions, probabilities = predict_churn(df)
        proba = probabilities[0]
        
        with col_result:
            st.write("Hasil Simulasi")
            
            baseline = 0.26 
            diff = proba - baseline
            
            st.metric(
                label="Churn Probability",
                value=f"{proba:.1%}",
                delta=f"{diff:+.1%}",
                delta_color="inverse"
            )

            if proba > 0.7:
                st.error("High Risk: Strategi berisiko tinggi.")
            elif proba > 0.4:
                st.warning("Medium Risk: Perlu optimasi.")
            else:
                st.success("Low Risk: Strategi efektif.")

    except Exception as e:
        st.error(f"Error: {e}")
