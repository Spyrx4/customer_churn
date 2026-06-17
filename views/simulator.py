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
        if st.button("Promo Paket Hemat", width='stretch', help="Diskon 20% Monthly Charges"):
            st.session_state["sim_monthly"] = float(sim_data.get('MonthlyCharges', 70.0)) * 0.8
        if st.button("Loyalty Lock", width='stretch', help="Ubah kontrak ke Two Year"):
            st.session_state["sim_contract"] = "Two year"
        if st.button("Tech-Security Bundle", width='stretch', help="Aktifkan Tech Support & Online Security"):
            if sim_data.get('InternetService', 'No') != 'No':
                st.session_state["sim_tech"] = "Yes"
                st.session_state["sim_sec"] = "Yes"
            
        if st.button("Reset Simulator", type="secondary", width='stretch'):
            for k in ["sim_monthly", "sim_contract", "sim_tech", "sim_sec", "sim_tenure"]:
                st.session_state.pop(k, None)
            st.rerun()

    with col_input:
        st.write("Manual Input Strategi")
        with st.form("simulator_form"):
            default_monthly = float(sim_data.get('MonthlyCharges', 70.0))
            default_contract = sim_data.get('Contract', 'Month-to-month')
            default_tech = sim_data.get('TechSupport', 'No')
            default_sec = sim_data.get('OnlineSecurity', 'No')
            default_tenure = int(sim_data.get('tenure', 12))

            new_monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=float(default_monthly), key="sim_monthly")
            new_contract = st.selectbox("Contract Type", contract_opts, index=get_idx(contract_opts, default_contract), key="sim_contract")
            new_tech = st.selectbox("Tech Support", yes_no_inet, index=get_idx(yes_no_inet, default_tech), key="sim_tech")
            new_sec = st.selectbox("Online Security", yes_no_inet, index=get_idx(yes_no_inet, default_sec), key="sim_sec")
            new_tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=72, value=int(default_tenure), key="sim_tenure")
            
            submitted = st.form_submit_button("Simulasikan", type="primary", use_container_width=True)

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
