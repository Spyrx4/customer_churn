import streamlit as st
import pandas as pd
import numpy as np

def inject_simulator():
    st.markdown("---")
    st.markdown("### Churn Prevention Simulator")
    st.markdown("Ubah variabel di bawah untuk simulasi strategi bisnis atau terapkan *Strategy Presets*.")

    # Initialize from session state or default
    sim_data = st.session_state.get("sim_data", {})
    if not sim_data:
        sim_data = {
            'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': 12, 'PhoneService': 'Yes', 'MultipleLines': 'No',
            'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
            'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check', 'MonthlyCharges': 70.0, 'TotalCharges': 840.0
        }

    # Helper function to get index safely
    def get_idx(options, val):
        return options.index(val) if val in options else 0

    contract_opts = ["Month-to-month", "One year", "Two year"]
    yes_no_inet = ["Yes", "No", "No internet service"]

    col_presets, col_input, col_result = st.columns([1, 1, 1])

    with col_presets:
        st.write("Strategy Presets")
        if st.button("Promo Paket Hemat", use_container_width=True, help="Diskon 20% Monthly Charges"):
            st.session_state["sim_promo_hemat"] = True
        if st.button("Loyalty Lock", use_container_width=True, help="Ubah kontrak ke Two Year"):
            st.session_state["sim_loyalty_lock"] = True
        if st.button("Tech-Security Bundle", use_container_width=True, help="Aktifkan Tech Support & Online Security"):
            st.session_state["sim_tech_sec"] = True
            
        if st.button("Reset Simulator", type="secondary", use_container_width=True):
            st.session_state.pop("sim_promo_hemat", None)
            st.session_state.pop("sim_loyalty_lock", None)
            st.session_state.pop("sim_tech_sec", None)
            st.rerun()

    with col_input:
        st.write("Manual Input Strategi")
        
        # Apply presets directly to default UI values if triggered
        default_monthly = float(sim_data.get('MonthlyCharges', 70.0))
        if st.session_state.get("sim_promo_hemat"):
            default_monthly *= 0.8
            
        default_contract = sim_data.get('Contract', 'Month-to-month')
        if st.session_state.get("sim_loyalty_lock"):
            default_contract = 'Two year'
            
        default_tech = sim_data.get('TechSupport', 'No')
        default_sec = sim_data.get('OnlineSecurity', 'No')
        if st.session_state.get("sim_tech_sec"):
            if sim_data.get('InternetService', 'No') != 'No':
                default_tech = 'Yes'
                default_sec = 'Yes'

        new_monthly = st.slider("Monthly Charges ($)", 0.0, 150.0, float(default_monthly), key="sim_monthly")
        new_contract = st.selectbox("Contract Type", contract_opts, index=get_idx(contract_opts, default_contract), key="sim_contract")
        new_tech = st.selectbox("Tech Support", yes_no_inet, index=get_idx(yes_no_inet, default_tech), key="sim_tech")
        new_sec = st.selectbox("Online Security", yes_no_inet, index=get_idx(yes_no_inet, default_sec), key="sim_sec")
        new_tenure = st.slider("Tenure (bulan)", 0, 72, int(sim_data.get('tenure', 12)), key="sim_tenure")

    # Construct dataframe for BEFORE
    raw_before = pd.DataFrame([sim_data])
    
    # Construct dataframe for AFTER
    after_data = sim_data.copy()
    after_data.update({
        'MonthlyCharges': new_monthly,
        'Contract': new_contract,
        'TechSupport': new_tech,
        'OnlineSecurity': new_sec,
        'tenure': new_tenure,
        'TotalCharges': new_monthly * new_tenure if new_tenure > 0 else new_monthly
    })
    raw_after = pd.DataFrame([after_data])

    try:
        from utils.ml_utils import predict_churn
        
        # Inference Before
        _, prob_before_arr = predict_churn(raw_before)
        prob_before = prob_before_arr[0]
        
        # Inference After
        _, prob_after_arr = predict_churn(raw_after)
        prob_after = prob_after_arr[0]
        
        with col_result:
            st.write("Comparison Analytics")
            
            diff = prob_after - prob_before
            
            c1, c2 = st.columns(2)
            c1.metric(label="Probability Before", value=f"{prob_before:.1%}")
            c2.metric(
                label="Probability After",
                value=f"{prob_after:.1%}",
                delta=f"{diff:+.1%}",
                delta_color="inverse"
            )

            st.markdown("---")
            if prob_after > 0.7:
                st.error("High Risk: Strategi berisiko tinggi.")
            elif prob_after > 0.4:
                st.warning("Medium Risk: Perlu optimasi.")
            else:
                st.success("Low Risk: Strategi efektif.")

    except Exception as e:
        st.error(f"Error: {e}")
