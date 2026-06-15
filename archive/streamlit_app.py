# churn prediction form (standalone)

import streamlit as st
import requests

st.set_page_config(page_title="Churn Prediction", layout='centered')
st.title("Customer Churn Prediction")

yes_no = ["Yes", "No"]
m_f = ["Male", "Female"]

def yn(x):
    return 1 if x == "Yes" else 0


st.subheader("Customer Profile")
gender = st.selectbox("Gender", m_f)
seniorCtzn = int(st.selectbox("Senior Citizen", options=[0,1], format_func=lambda x: "Yes" if x == 1 else "No"))
partner = st.selectbox("Partner", yes_no)
dependents = st.selectbox("Dependents", yes_no)
tenure = st.slider("Tenure", 0, 60, 12)

st.subheader("Add-on Services")
phoneService = st.selectbox("Phone Service", yes_no)
multipleLines = st.selectbox("Multiplelines", ['No', 'Yes', 'No phone service'])
internetservice = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
onlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
onlineBackup = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
deviceProtect = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
techSupp = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
streamingTv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
streamingMov = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
contract = st.selectbox("Contract", ['One year', 'Two year', 'Month-to-month'])

st.subheader("Payment")
paperlessBill = st.selectbox("Paperless Billing", yes_no)
payMeth = st.selectbox("Payment Method", ['Mailed check', 'Credit card (automatic)', 'Electronic check', 'Bank transfer (automatic)'])
monthCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=120.0, value=74.0)
totalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10_000.0, value=1_433.0)

if st.button("Predict Churn"):
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
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )
        
        result = response.json()
        
        st.subheader("Prediction result")
        st.write(f"**Churn Probability:** {round(result['churn_probability'], 3)}")
        st.write(f"**Risk Level** {result['risk_level']}")
        
        if result["prediction"] == 1:
            st.error("Customer likely to churn")
        else:
            st.success("Customer likely to stay")
            
    except Exception as e:
        st.error("Could not connect to FastAPI Backend")
        st.write("Make sure FastAPI is running at http://127.0.0.1:8000")
        st.write(str(e))
