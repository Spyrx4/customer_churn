import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

def render_analytics(filtered, churn_rate):
    st.markdown("---")

    # kpi
    churn_count = (filtered["Churn"] == "Yes").sum()
    total_cust = len(filtered)
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

    # top churn drivers
    st.markdown('<p class="dashboard-subheader">Top Churn Drivers</p>', unsafe_allow_html=True)
    if churn_count > 0:
        def get_highest_churn(col_name):
            return filtered.groupby(col_name)["Churn"].apply(lambda x: (x == "Yes").mean()).idxmax(), \
                   filtered.groupby(col_name)["Churn"].apply(lambda x: (x == "Yes").mean()).max()

        h_contract, p_contract = get_highest_churn("Contract")
        h_pay, p_pay = get_highest_churn("PaymentMethod")
        h_inet, p_inet = get_highest_churn("InternetService")

        c1, c2, c3 = st.columns(3)
        c1.info(f"**Contract:** {h_contract} ({p_contract:.1%})")
        c2.info(f"**Payment:** {h_pay} ({p_pay:.1%})")
        c3.info(f"**Internet:** {h_inet} ({p_inet:.1%})")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate Strategic Recommendations", type="primary", use_container_width=True):
            with st.spinner("AI Consultant sedang menyusun strategi..."):
                from llm.agent import run_agent
                prompt = (
                    f"Berdasarkan data saat ini, faktor penyebab churn tertinggi adalah: "
                    f"Contract: {h_contract} ({p_contract:.1%}), Payment Method: {h_pay} ({p_pay:.1%}), "
                    f"dan Internet Service: {h_inet} ({p_inet:.1%}). "
                    f"Berikan rekomendasi strategis bisnis tingkat makro untuk menekan churn pada kelompok ini. "
                    f"Format dalam poin-poin yang profesional tanpa emoji."
                )
                try:
                    ai_response, _ = run_agent(prompt, [])
                    st.success("Strategi Berhasil Dibuat!")
                    st.markdown(ai_response)
                except Exception as e:
                    st.error(f"Gagal memuat rekomendasi: {e}")
    else:
        st.info("No churn data available for current filter.")

    st.markdown("---")

    # churn distribution + contract
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="dashboard-subheader">Distribusi Churn</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="dashboard-subheader">Churn Rate by Contract Type</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="dashboard-subheader">Churn by Internet Service</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="dashboard-subheader">Churn by Payment Method</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="dashboard-subheader">Tenure Cohorts vs Churn</p>', unsafe_allow_html=True)
        
        bins = [-1, 12, 36, 60, 1000]
        labels = ["New (0-1 yr)", "Regular (1-3 yrs)", "Loyal (3-5 yrs)", "Very Loyal (>5 yrs)"]
        filtered_cohorts = filtered.copy()
        filtered_cohorts["TenureGroup"] = pd.cut(filtered_cohorts["tenure"], bins=bins, labels=labels)
        
        cohort_churn = (
            filtered_cohorts.groupby("TenureGroup", observed=False)["Churn"]
            .value_counts(normalize=True).rename("Proportion").reset_index()
        )
        
        fig_tenure = px.bar(
            cohort_churn, x="TenureGroup", y="Proportion", color="Churn",
            barmode="group", color_discrete_map=CHURN_COLORS,
            text=cohort_churn["Proportion"].apply(lambda x: f"{x:.1%}") if not cohort_churn.empty else None
        )
        fig_tenure.update_traces(textposition="outside", textfont_size=11)
        fig_tenure.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_tickformat=".0%",
                                  xaxis_title="", yaxis_title="Proporsi")
        st.plotly_chart(fig_tenure, use_container_width=True)

    with col6:
        st.markdown('<p class="dashboard-subheader">Monthly Charges by Churn</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="dashboard-subheader">Churn Rate per Add-on Service</p>', unsafe_allow_html=True)
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
            text=[[f"{v:.1%}"for v in row] for row in svc_pivot.values],
            texttemplate="%{text}", textfont=dict(size=13, color="white"),
            hovertemplate="Service: %{y}<br>Status: %{x}<br>Churn Rate: %{text}<extra></extra>",
            colorbar=dict(title=dict(text="Churn Rate", font=dict(color="#cbd5e1")),
                          tickformat=".0%", tickfont=dict(color="#cbd5e1")),
        ))
        fig_heatmap.update_layout(**PLOTLY_LAYOUT, height=380,
                                   xaxis_title="Berlangganan Service?", yaxis_title="")
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col8:
        st.markdown('<p class="dashboard-subheader">Senior Citizen vs Churn</p>', unsafe_allow_html=True)
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
    st.markdown('<p class="dashboard-subheader">Tenure vs Total Charges (Sampled)</p>', unsafe_allow_html=True)
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
