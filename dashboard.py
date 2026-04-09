# dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import batch_predict

# page config
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# css
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
    }
    div[data-testid="stMetric"] label {
        color: #a5b4fc !important;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Plotly chart containers */
    div[data-testid="stPlotlyChart"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 8px;
    }

    /* Headers */
    h1, h2, h3 {
        color: #e0e7ff !important;
    }

    /* Divider */
    hr {
        border-color: rgba(165, 180, 252, 0.2) !important;
    }

    .dashboard-subheader {
        color: #a5b4fc;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    /* Prediction result cards */
    .pred-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    .pred-card h2 {
        margin: 0 0 4px 0;
        font-size: 36px;
    }
    .pred-card p {
        color: #94a3b8;
        margin: 0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-high { border-left: 4px solid #f43f5e; }
    .risk-medium { border-left: 4px solid #fb923c; }
    .risk-low { border-left: 4px solid #34d399; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(255,255,255,0.08);
        color: #a5b4fc;
        padding: 10px 24px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        border-bottom: 2px solid #6366f1 !important;
        color: #e0e7ff !important;
    }

    /* Chat styling */
    .agent-greeting {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
    }
    .agent-greeting h3 {
        color: #a5b4fc !important;
        margin: 0 0 8px 0;
    }
    .agent-greeting p {
        color: #94a3b8;
        margin: 0;
        font-size: 14px;
        line-height: 1.6;
    }
    .quick-prompt-btn button {
        background: rgba(99, 102, 241, 0.12) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        color: #a5b4fc !important;
        border-radius: 20px !important;
        font-size: 13px !important;
        padding: 6px 16px !important;
        transition: all 0.2s ease !important;
    }
    .quick-prompt-btn button:hover {
        background: rgba(99, 102, 241, 0.25) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    .tool-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        font-size: 11px;
        padding: 2px 10px;
        border-radius: 10px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    df = df.drop(columns=["id"], errors="ignore")
    return df


df = load_data()

# sidebar filters
st.sidebar.markdown("## 🎛️ Dashboard Filters")

gender_filter = st.sidebar.multiselect(
    "Gender", options=df["gender"].unique(), default=df["gender"].unique()
)
contract_filter = st.sidebar.multiselect(
    "Contract", options=df["Contract"].unique(), default=df["Contract"].unique()
)
internet_filter = st.sidebar.multiselect(
    "Internet Service",
    options=df["InternetService"].unique(),
    default=df["InternetService"].unique(),
)
senior_filter = st.sidebar.multiselect(
    "Senior Citizen",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)

mask = (
    df["gender"].isin(gender_filter)
    & df["Contract"].isin(contract_filter)
    & df["InternetService"].isin(internet_filter)
    & df["SeniorCitizen"].isin(senior_filter)
)
filtered = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing:** {len(filtered):,} / {len(df):,} customers")

# color palette
CHURN_COLORS = {"Yes": "#f43f5e", "No": "#6366f1"}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
    ),
)

# header
st.markdown("# 📊 Customer Churn Analytics")
st.markdown(
    '<p style="color:#94a3b8; margin-top:-10px;">Dashboard analitik & prediksi churn pelanggan</p>',
    unsafe_allow_html=True,
)

# tabs
tab_dashboard, tab_predict, tab_batch, tab_agent = st.tabs(["📊 Dashboard Analytics", "🔮 Prediksi Churn", "📂 Batch Prediction", "🤖 AI Consultant"])

# tab 1 - analytics
with tab_dashboard:
    st.markdown("---")

    # kpi
    churn_count = (filtered["Churn"] == "Yes").sum()
    total_cust = len(filtered)
    churn_rate = churn_count / total_cust * 100 if total_cust > 0 else 0
    avg_monthly = filtered["MonthlyCharges"].mean()
    avg_tenure = filtered["tenure"].mean()
    avg_total = filtered["TotalCharges"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Customers", f"{total_cust:,}")
    k2.metric("Churn Rate", f"{churn_rate:.1f}%")
    k3.metric("Avg Monthly Charges", f"${avg_monthly:,.1f}")
    k4.metric("Avg Tenure", f"{avg_tenure:.0f} bulan")
    k5.metric("Avg Total Charges", f"${avg_total:,.0f}")

    st.markdown("---")

    # churn distribution + contract
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="dashboard-subheader">📌 Distribusi Churn</p>', unsafe_allow_html=True)
        churn_dist = filtered["Churn"].value_counts().reset_index()
        churn_dist.columns = ["Churn", "Count"]
        fig_donut = px.pie(
            churn_dist, values="Count", names="Churn", hole=0.55,
            color="Churn", color_discrete_map=CHURN_COLORS,
        )
        fig_donut.update_traces(
            textinfo="percent+label", textfont_size=14,
            marker=dict(line=dict(color="#1a1a3e", width=2)),
        )
        fig_donut.update_layout(
            **PLOTLY_LAYOUT, showlegend=False, height=380,
            annotations=[dict(
                text=f"<b>{churn_rate:.1f}%</b><br>Churn",
                x=0.5, y=0.5, font_size=18, font_color="#f43f5e", showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown('<p class="dashboard-subheader">📋 Churn Rate by Contract Type</p>', unsafe_allow_html=True)
        contract_churn = (
            filtered.groupby("Contract")["Churn"]
            .value_counts(normalize=True).rename("Proportion").reset_index()
        )
        fig_contract = px.bar(
            contract_churn, x="Contract", y="Proportion", color="Churn",
            barmode="group", color_discrete_map=CHURN_COLORS,
            text=contract_churn["Proportion"].apply(lambda x: f"{x:.1%}"),
        )
        fig_contract.update_traces(textposition="outside", textfont_size=11)
        fig_contract.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_tickformat=".0%",
                                    xaxis_title="", yaxis_title="Proporsi")
        st.plotly_chart(fig_contract, use_container_width=True)

    # internet service + payment method
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="dashboard-subheader">🌐 Churn by Internet Service</p>', unsafe_allow_html=True)
        inet_churn = (
            filtered.groupby("InternetService")["Churn"]
            .value_counts(normalize=True).rename("Proportion").reset_index()
        )
        fig_inet = px.bar(
            inet_churn, x="InternetService", y="Proportion", color="Churn",
            barmode="stack", color_discrete_map=CHURN_COLORS,
            text=inet_churn["Proportion"].apply(lambda x: f"{x:.1%}"),
        )
        fig_inet.update_traces(textposition="inside", textfont_size=11)
        fig_inet.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_tickformat=".0%",
                                xaxis_title="", yaxis_title="Proporsi")
        st.plotly_chart(fig_inet, use_container_width=True)

    with col4:
        st.markdown('<p class="dashboard-subheader">💳 Churn by Payment Method</p>', unsafe_allow_html=True)
        pay_churn = (
            filtered.groupby("PaymentMethod")["Churn"]
            .apply(lambda x: (x == "Yes").mean())
            .sort_values(ascending=True).reset_index()
        )
        pay_churn.columns = ["PaymentMethod", "ChurnRate"]
        fig_pay = px.bar(
            pay_churn, y="PaymentMethod", x="ChurnRate", orientation="h",
            text=pay_churn["ChurnRate"].apply(lambda x: f"{x:.1%}"),
            color="ChurnRate", color_continuous_scale=["#6366f1", "#f43f5e"],
        )
        fig_pay.update_traces(textposition="outside", textfont_size=12)
        fig_pay.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_tickformat=".0%",
                               xaxis_title="Churn Rate", yaxis_title="",
                               coloraxis_showscale=False)
        st.plotly_chart(fig_pay, use_container_width=True)

    st.markdown("---")

    # tenure + monthly charges
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<p class="dashboard-subheader">⏳ Distribusi Tenure by Churn</p>', unsafe_allow_html=True)
        fig_tenure = px.histogram(
            filtered, x="tenure", color="Churn", nbins=36,
            barmode="overlay", color_discrete_map=CHURN_COLORS, opacity=0.75,
        )
        fig_tenure.update_layout(**PLOTLY_LAYOUT, height=380,
                                  xaxis_title="Tenure (bulan)", yaxis_title="Jumlah Customer")
        st.plotly_chart(fig_tenure, use_container_width=True)

    with col6:
        st.markdown('<p class="dashboard-subheader">💰 Monthly Charges by Churn</p>', unsafe_allow_html=True)
        fig_box = px.box(
            filtered, x="Churn", y="MonthlyCharges", color="Churn",
            color_discrete_map=CHURN_COLORS,
        )
        fig_box.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                               xaxis_title="", yaxis_title="Monthly Charges ($)")
        st.plotly_chart(fig_box, use_container_width=True)

    # heatmap + senior citizen
    col7, col8 = st.columns(2)

    with col7:
        st.markdown('<p class="dashboard-subheader">🔥 Churn Rate per Add-on Service</p>', unsafe_allow_html=True)
        services = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        svc_data = []
        for svc in services:
            for val in filtered[svc].unique():
                if val == "No internet service":
                    continue
                subset = filtered[filtered[svc] == val]
                rate = (subset["Churn"] == "Yes").mean()
                svc_data.append({"Service": svc, "Status": val, "ChurnRate": rate})

        svc_df = pd.DataFrame(svc_data)
        svc_pivot = svc_df.pivot(index="Service", columns="Status", values="ChurnRate")
        col_order = [c for c in ["No", "Yes"] if c in svc_pivot.columns]
        svc_pivot = svc_pivot[col_order]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=svc_pivot.values, x=svc_pivot.columns.tolist(), y=svc_pivot.index.tolist(),
            colorscale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#f43f5e"]],
            text=[[f"{v:.1%}" for v in row] for row in svc_pivot.values],
            texttemplate="%{text}", textfont=dict(size=13, color="white"),
            hovertemplate="Service: %{y}<br>Status: %{x}<br>Churn Rate: %{text}<extra></extra>",
            colorbar=dict(title=dict(text="Churn Rate", font=dict(color="#cbd5e1")),
                          tickformat=".0%", tickfont=dict(color="#cbd5e1")),
        ))
        fig_heatmap.update_layout(**PLOTLY_LAYOUT, height=380,
                                   xaxis_title="Berlangganan Service?", yaxis_title="")
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col8:
        st.markdown('<p class="dashboard-subheader">👴 Senior Citizen vs Churn</p>', unsafe_allow_html=True)
        senior_churn = (
            filtered.groupby("SeniorCitizen")["Churn"]
            .value_counts(normalize=True).rename("Proportion").reset_index()
        )
        senior_churn["SeniorCitizen"] = senior_churn["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
        fig_senior = px.bar(
            senior_churn, x="SeniorCitizen", y="Proportion", color="Churn",
            barmode="group", color_discrete_map=CHURN_COLORS,
            text=senior_churn["Proportion"].apply(lambda x: f"{x:.1%}"),
        )
        fig_senior.update_traces(textposition="outside", textfont_size=12)
        fig_senior.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_tickformat=".0%",
                                  xaxis_title="", yaxis_title="Proporsi")
        st.plotly_chart(fig_senior, use_container_width=True)

    st.markdown("---")

    # scatter plot
    st.markdown('<p class="dashboard-subheader">📈 Tenure vs Total Charges (Sampled)</p>', unsafe_allow_html=True)
    sample_size = min(5000, len(filtered))
    sampled = filtered.sample(n=sample_size, random_state=42)
    fig_scatter = px.scatter(
        sampled, x="tenure", y="TotalCharges", color="Churn", size="MonthlyCharges",
        color_discrete_map=CHURN_COLORS, opacity=0.6,
        hover_data=["Contract", "InternetService", "PaymentMethod"],
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=450,
                               xaxis_title="Tenure (bulan)", yaxis_title="Total Charges ($)")
    st.plotly_chart(fig_scatter, use_container_width=True)


# tab 2 - churn prediction
with tab_predict:
    st.markdown("---")
    st.markdown(
        '<p style="color:#94a3b8;">Isi data pelanggan di bawah ini, lalu klik <b>Predict</b> '
        'untuk melihat kemungkinan churn.</p>',
        unsafe_allow_html=True,
    )

    yes_no = ["Yes", "No"]

    # customer profile
    st.markdown("### 👤 Customer Profile")
    cp1, cp2, cp3, cp4, cp5 = st.columns(5)
    with cp1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
    with cp2:
        seniorCtzn = int(st.selectbox("Senior Citizen", options=[0, 1],
                                       format_func=lambda x: "Yes" if x == 1 else "No",
                                       key="pred_senior"))
    with cp3:
        partner = st.selectbox("Partner", yes_no, key="pred_partner")
    with cp4:
        dependents = st.selectbox("Dependents", yes_no, key="pred_dep")
    with cp5:
        tenure = st.slider("Tenure (bulan)", 0, 72, 12, key="pred_tenure")

    # services
    st.markdown("### 📡 Services")
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

    # contract & payment
    st.markdown("### 💳 Contract & Payment")
    py1, py2, py3, py4 = st.columns(4)
    with py1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="pred_contract")
    with py2:
        paperlessBill = st.selectbox("Paperless Billing", yes_no, key="pred_paper")
    with py3:
        payMeth = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Credit card (automatic)", "Bank transfer (automatic)",
        ], key="pred_pay")
    with py4:
        monthCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=120.0,
                                        value=74.0, key="pred_monthly")

    totalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10_000.0,
                                    value=1_433.0, key="pred_total")

    st.markdown("---")

    # predict
    if st.button("🔮  Predict Churn", use_container_width=True, type="primary"):
        raw = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': seniorCtzn,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phoneService,
            'MultipleLines': multipleLines,
            'InternetService': internetservice,
            'OnlineSecurity': onlineSecurity,
            'OnlineBackup': onlineBackup,
            'DeviceProtection': deviceProtect,
            'TechSupport': techSupp,
            'StreamingTV': streamingTv,
            'StreamingMovies': streamingMov,
            'Contract': contract,
            'PaperlessBilling': paperlessBill,
            'PaymentMethod': payMeth,
            'MonthlyCharges': monthCharges,
            'TotalCharges': totalCharges,
        }])

        try:
            predictions, probabilities = batch_predict._batch_inference(raw)
            pred = int(predictions[0])
            proba = float(probabilities[0])
            risk = (
                "High" if proba >= 0.8 else
                "Medium" if proba >= 0.5 else
                "Low"
            )

            risk_css = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}

            st.markdown("---")
            st.markdown("### 📋 Hasil Prediksi")

            r1, r2, r3 = st.columns(3)
            with r1:
                color = "#f43f5e" if pred == 1 else "#34d399"
                label = "CHURN" if pred == 1 else "STAY"
                st.markdown(
                    f'<div class="pred-card {risk_css.get(risk, "")}">'
                    f'<h2 style="color:{color}">{label}</h2>'
                    f'<p>Prediksi</p></div>',
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f'<div class="pred-card {risk_css.get(risk, "")}">'
                    f'<h2 style="color:#e0e7ff">{proba:.1%}</h2>'
                    f'<p>Probabilitas Churn</p></div>',
                    unsafe_allow_html=True,
                )
            with r3:
                risk_color = {"High": "#f43f5e", "Medium": "#fb923c", "Low": "#34d399"}
                st.markdown(
                    f'<div class="pred-card {risk_css.get(risk, "")}">'
                    f'<h2 style="color:{risk_color.get(risk, "#e0e7ff")}">{risk}</h2>'
                    f'<p>Risk Level</p></div>',
                    unsafe_allow_html=True,
                )

            # gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
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
                    threshold=dict(
                        line=dict(color="#f43f5e", width=3),
                        thickness=0.8, value=50,
                    ),
                ),
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error("⚠️ Terjadi kesalahan saat prediksi")
            st.code(str(e), language="text")

# tab 3 - batch prediction
with tab_batch:
    batch_predict.render()

# tab 4 - AI Agent Consultant
with tab_agent:
    st.markdown("---")

    # greeting
    st.markdown(
        '<div class="agent-greeting">'
        '<h3>🤖 Rini — AI Business Consultant</h3>'
        '<p>Halo! Saya <b>Rini</b>, AI Business Consultant Nusantara Connect. '
        'Saya bisa membantu Anda:</p>'
        '<p>'
        '🔍 <b>Menjawab pertanyaan</b> tentang perusahaan &amp; layanan<br>'
        '🔮 <b>Memprediksi churn</b> pelanggan secara langsung<br>'
        '📊 <b>Menganalisis data</b> pelanggan (594K records)<br>'
        '📈 <b>Membuat visualisasi</b> chart interaktif'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # session state init
    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []  # [{role, content, charts?}]
    if "agent_llm_history" not in st.session_state:
        st.session_state["agent_llm_history"] = []  # clean role/content for API

    # quick prompts
    st.markdown('<p style="color:#64748b; font-size:13px; margin-bottom:4px;">💡 Contoh pertanyaan:</p>', unsafe_allow_html=True)
    qp_cols = st.columns(4)
    quick_prompts = [
        "Berapa churn rate pelanggan Fiber optic?",
        "Prediksi churn: Female, tenure 3, Fiber optic, Month-to-month",
        "Buatkan grafik churn rate by contract type",
        "Apa layanan utama Nusantara Connect?",
    ]
    selected_prompt = None
    for i, qp in enumerate(quick_prompts):
        with qp_cols[i]:
            st.markdown('<div class="quick-prompt-btn">', unsafe_allow_html=True)
            if st.button(qp, key=f"qp_{i}", use_container_width=True):
                selected_prompt = qp
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # render chat history
    for msg in st.session_state["agent_history"]:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            # render charts if present
            if msg.get("charts"):
                for fig in msg["charts"]:
                    st.plotly_chart(fig, use_container_width=True)

    # chat input
    user_input = st.chat_input("Tanya Rini tentang data, prediksi, atau analisis...", key="agent_chat_input")

    # use quick prompt if selected
    if selected_prompt:
        user_input = selected_prompt

    if user_input:
        # display user message
        st.session_state["agent_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # call agent
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔄 Rini sedang berpikir..."):
                try:
                    from llm.agent import run_agent
                    response_text, charts = run_agent(
                        user_input,
                        st.session_state["agent_llm_history"],
                    )
                except Exception as e:
                    response_text = f"⚠️ Terjadi kesalahan: {e}"
                    charts = []

            st.markdown(response_text)
            for fig in charts:
                st.plotly_chart(fig, use_container_width=True)

        # save to history
        st.session_state["agent_history"].append({
            "role": "assistant",
            "content": response_text,
            "charts": charts,
        })
        # LLM history (no charts, just text)
        st.session_state["agent_llm_history"].append({"role": "user", "content": user_input})
        st.session_state["agent_llm_history"].append({"role": "assistant", "content": response_text})

        # keep history manageable (last 20 turns)
        if len(st.session_state["agent_llm_history"]) > 40:
            st.session_state["agent_llm_history"] = st.session_state["agent_llm_history"][-40:]

        st.rerun()

    # clear chat button
    if st.session_state["agent_history"]:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", key="clear_agent_chat", use_container_width=True):
            st.session_state["agent_history"] = []
            st.session_state["agent_llm_history"] = []
            st.rerun()

# footer
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#64748b; font-size:13px;">'
    "Customer Churn Dashboard &bull; Data: train.csv (594K rows) &bull; "
    "Model: XGBoost + Random OverSampling</p>",
    unsafe_allow_html=True,
)
