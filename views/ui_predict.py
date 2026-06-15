import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_prediction():
    st.markdown("---")
    st.markdown(
        '<p style="color:#94a3b8;">Isi data pelanggan di bawah ini untuk melihat prediksi churn.</p>',
        unsafe_allow_html=True,
    )
    yes_no = ["Yes", "No"]

    # UI Components (Data Input)
    st.markdown("### Customer Profile")
    cp1, cp2, cp3, cp4, cp5 = st.columns(5)
    with cp1: gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
    with cp2: seniorCtzn = int(st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="pred_senior"))
    with cp3: partner = st.selectbox("Partner", yes_no, key="pred_partner")
    with cp4: dependents = st.selectbox("Dependents", yes_no, key="pred_dep")
    with cp5: tenure = st.slider("Tenure (bulan)", 0, 72, 12, key="pred_tenure")

    st.markdown("### Services")
    sv1, sv2, sv3 = st.columns(3)
    with sv1:
        phoneService = st.selectbox("Phone Service", yes_no, key="pred_phone")
        multipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="pred_multi")
        internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="pred_inet")
    with sv2:
        onlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="pred_sec")
        onlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="pred_bkp")
        deviceProtect = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="pred_dev")
    with sv3:
        techSupp = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="pred_tech")
        streamingTv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="pred_tv")
        streamingMov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="pred_mov")

    st.markdown("### Contract & Payment")
    py1, py2, py3, py4 = st.columns(4)
    with py1: contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="pred_contract")
    with py2: paperlessBill = st.selectbox("Paperless Billing", yes_no, key="pred_paper")
    with py3: payMeth = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)"], key="pred_pay")
    with py4: monthCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=120.0, value=74.0, key="pred_monthly")

    totalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1433.0, key="pred_total")

    st.markdown("---")

    if st.button("🚀 Predict Churn", use_container_width=True, type="primary"):
        # Payload prediksi
        payload = {
            "gender": gender,
            "SeniorCitizen": seniorCtzn,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phoneService,
            "MultipleLines": multipleLines,
            "InternetService": internetservice,
            "OnlineSecurity": onlineSecurity,
            "OnlineBackup": onlineBackup,
            "DeviceProtection": deviceProtect,
            "TechSupport": techSupp,
            "StreamingTV": streamingTv,
            "StreamingMovies": streamingMov,
            "Contract": contract,
            "PaperlessBilling": paperlessBill,
            "PaymentMethod": payMeth,
            "MonthlyCharges": monthCharges,
            "TotalCharges": totalCharges
        }

        try:
            with st.spinner("Menganalisis data pelanggan..."):
                from utils.ml_utils import predict_churn
                raw_df = pd.DataFrame([payload])
                
                predictions, probabilities = predict_churn(raw_df)
                pred = int(predictions[0])
                proba = float(probabilities[0])
                risk = "High" if proba >= 0.8 else "Medium" if proba >= 0.5 else "Low"
                
            st.success("✅ Prediksi Berhasil")
            
            # TAMPILIN HASIL
            r1, r2, r3 = st.columns(3)
            risk_css = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}
            risk_color = {"High": "#f43f5e", "Medium": "#fb923c", "Low": "#34d399"}
            
            with r1:
                color = "#f43f5e" if pred == 1 else "#34d399"
                st.markdown(f'<div class="pred-card {risk_css.get(risk, "")}"><h2 style="color:{color}">{"CHURN" if pred == 1 else "STAY"}</h2><p>Prediksi</p></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="pred-card {risk_css.get(risk, "")}"><h2 style="color:#e0e7ff">{proba:.1%}</h2><p>Probabilitas</p></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="pred-card {risk_css.get(risk, "")}"><h2 style="color:{risk_color.get(risk, "#e0e7ff")}">{risk}</h2><p>Risk Level</p></div>', unsafe_allow_html=True)

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                number=dict(suffix="%", font=dict(color="#e0e7ff", size=48)),
                title=dict(text="Churn Probability", font=dict(color="#a5b4fc", size=16)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(color="#64748b")),
                    bar=dict(color="#6366f1"),
                    bgcolor="rgba(255,255,255,0.05)",
                    steps=[
                        dict(range=[0, 50], color="rgba(99, 102, 241, 0.15)"),
                        dict(range=[50, 80], color="rgba(251, 146, 60, 0.2)"),
                        dict(range=[80, 100], color="rgba(244, 63, 94, 0.25)"),
                    ],
                )
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"), height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # AI Recommendation for individual customer
            st.markdown("---")
            st.markdown("### 🤖 Rekomendasi Tindakan (AI Prescriptive Analytics)")
            with st.spinner("Menghasilkan rekomendasi retensi personal..."):
                from llm.agent import run_agent
                profile_str = (
                    f"Contract: {contract}, Tenure: {tenure} bulan, Monthly Charges: ${monthCharges}, "
                    f"Internet Service: {internetservice}, Tech Support: {techSupp}, Payment: {payMeth}"
                )
                prompt = (
                    f"Customer dengan profil berikut diprediksi memiliki probabilitas churn {proba:.1%} (Risiko {risk}).\n"
                    f"Profil: {profile_str}\n"
                    f"Berikan rekomendasi tindakan spesifik dan personal untuk agen customer service agar dapat mencegah churn pada customer ini. "
                    f"Format respons profesional tanpa emoji."
                )
                try:
                    ai_recommendation, _ = run_agent(prompt, [])
                    st.info(ai_recommendation)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memuat rekomendasi AI: {e}")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memprediksi churn")
            st.exception(e)
