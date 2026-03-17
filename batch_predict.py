# batch prediction module

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# required columns
REQUIRED_FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

SUPPORTED_FORMATS = {
    "csv": "CSV",
    "xlsx": "Excel (.xlsx)",
    "xls": "Excel (.xls)",
    "json": "JSON",
    "parquet": "Parquet",
    "tsv": "TSV (Tab-separated)",
}

# model & pipeline
TENURE_BINS = [0, 6, 12, 24, 60, np.inf]
TENURE_LABELS = list(range(len(TENURE_BINS) - 1))


@st.cache_resource
def _load_artifacts():
    preprocess = joblib.load("artifacts/preprocessing_fe/preprocessing_artifacts.joblib")
    model = joblib.load("artifacts/models/xgb_ros.pkl")
    return preprocess, model


def _batch_inference(raw_df: pd.DataFrame):
    preprocess, model = _load_artifacts()
    df = raw_df.copy()

    df["Tenure_bucket"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=False
    )
    for col in preprocess["log_cols"]:
        if col in df:
            df[col] = np.log1p(df[col])

    df[preprocess["cont_cols"]] = preprocess["scaler"].transform(df[preprocess["cont_cols"]])
    X = preprocess["ohe_preprocess"].transform(df)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        return pd.read_json(uploaded_file)
    elif name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Format file tidak didukung: {name}")


# plotly defaults
CHURN_COLORS = {"Yes": "#f43f5e", "No": "#6366f1", 1: "#f43f5e", 0: "#6366f1"}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
)


# render
def render():
    st.markdown("---")
    st.markdown(
        '<p style="color:#94a3b8;">'
        "Upload file data pelanggan untuk memprediksi churn secara massal. "
        "Model di-load langsung — tidak perlu FastAPI backend.</p>",
        unsafe_allow_html=True,
    )

    # format info
    with st.expander("📋 Format File & Kolom yang Diperlukan", expanded=False):
        st.markdown("**Format file yang didukung:**")
        fmt_cols = st.columns(6)
        for i, (ext, label) in enumerate(SUPPORTED_FORMATS.items()):
            fmt_cols[i].markdown(
                f'<span style="background:rgba(99,102,241,0.2); padding:4px 12px; '
                f'border-radius:8px; color:#a5b4fc; font-size:13px;">.{ext}</span>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown("**Kolom yang harus ada** (kolom `id` opsional, akan ditampilkan jika ada):")
        col_display = st.columns(4)
        for i, col_name in enumerate(REQUIRED_FEATURE_COLS):
            col_display[i % 4].markdown(f"- `{col_name}`")

    # upload
    uploaded_file = st.file_uploader(
        "📂 Upload file data pelanggan",
        type=list(SUPPORTED_FORMATS.keys()),
        help="Pastikan file memiliki kolom yang sesuai dengan data training",
        key="batch_uploader",
    )

    if uploaded_file is not None:
        try:
            raw_df = _read_uploaded_file(uploaded_file)
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            return

        has_id = "id" in raw_df.columns
        id_series = raw_df["id"].copy() if has_id else pd.RangeIndex(start=1, stop=len(raw_df) + 1)

        # validate columns
        missing_cols = [c for c in REQUIRED_FEATURE_COLS if c not in raw_df.columns]
        if missing_cols:
            st.error(f"❌ Kolom berikut tidak ditemukan: **{', '.join(missing_cols)}**")
            st.markdown("Kolom yang ditemukan dalam file:")
            st.code(", ".join(raw_df.columns.tolist()), language="text")
            return

        # preview
        st.markdown("---")
        st.markdown("### 📄 Preview Data")
        st.markdown(
            f'<p style="color:#94a3b8;">File: <b>{uploaded_file.name}</b> &bull; '
            f'{len(raw_df):,} baris &bull; {len(raw_df.columns)} kolom</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.markdown("---")

        # predict
        if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary", key="batch_btn"):
            with st.spinner("Memproses prediksi..."):
                feature_df = raw_df[REQUIRED_FEATURE_COLS].copy()
                try:
                    predictions, probabilities = _batch_inference(feature_df)
                except Exception as e:
                    st.error(f"❌ Error saat prediksi: {e}")
                    return

            result_df = pd.DataFrame()
            if has_id:
                result_df["id"] = id_series.values
            else:
                result_df["row_index"] = list(range(1, len(raw_df) + 1))

            result_df["Prediction"] = predictions
            result_df["Churn_Label"] = np.where(predictions == 1, "Yes", "No")
            result_df["Churn_Probability"] = np.round(probabilities, 4)
            result_df["Risk_Level"] = pd.cut(
                probabilities, bins=[0, 0.5, 0.8, 1.01],
                labels=["Low", "Medium", "High"], right=False,
            )
            for col in REQUIRED_FEATURE_COLS:
                result_df[col] = raw_df[col].values

            st.session_state["batch_result"] = result_df

        # results
        if "batch_result" in st.session_state:
            result_df = st.session_state["batch_result"]
            _render_results(result_df)
    else:
        st.markdown(
            '<div style="text-align:center; padding:80px 0;">'
            '<p style="font-size:48px; margin-bottom:8px;">📂</p>'
            '<p style="color:#64748b; font-size:16px;">'
            "Upload file untuk memulai prediksi batch</p>"
            '<p style="color:#475569; font-size:13px;">'
            f"Format yang didukung: {', '.join(f'.{ext}' for ext in SUPPORTED_FORMATS)}</p>"
            "</div>",
            unsafe_allow_html=True,
        )


# results section
def _render_results(result_df: pd.DataFrame):
    st.markdown("---")
    st.markdown("## 📊 Hasil Prediksi")

    total = len(result_df)
    churn_yes = (result_df["Prediction"] == 1).sum()
    churn_no = total - churn_yes
    churn_pct = churn_yes / total * 100 if total > 0 else 0
    avg_prob = result_df["Churn_Probability"].mean()
    high_risk = (result_df["Risk_Level"] == "High").sum()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Data", f"{total:,}")
    m2.metric("Predicted Churn", f"{churn_yes:,}")
    m3.metric("Predicted Stay", f"{churn_no:,}")
    m4.metric("Churn Rate", f"{churn_pct:.1f}%")
    m5.metric("Avg Probability", f"{avg_prob:.1%}")
    m6.metric("High Risk", f"{high_risk:,}")

    st.markdown("---")

    tab_table, tab_viz = st.tabs(["📋 Tabel Hasil", "📊 Visualisasi"])

    # table
    with tab_table:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            label_filter = st.multiselect(
                "Filter Prediksi", options=["Yes", "No"], default=["Yes", "No"],
                key="batch_label_filter",
            )
        with fc2:
            risk_filter = st.multiselect(
                "Filter Risk Level", options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"], key="batch_risk_filter",
            )
        with fc3:
            id_col = "id" if "id" in result_df.columns else "row_index"
            sort_col = st.selectbox(
                "Urutkan berdasarkan", options=["Churn_Probability", id_col],
                key="batch_sort",
            )
            sort_asc = st.checkbox("Ascending", value=False, key="batch_asc")

        display_df = result_df[
            result_df["Churn_Label"].isin(label_filter)
            & result_df["Risk_Level"].isin(risk_filter)
        ].sort_values(sort_col, ascending=sort_asc)

        st.markdown(
            f'<p style="color:#94a3b8;">Menampilkan {len(display_df):,} dari {total:,} baris</p>',
            unsafe_allow_html=True,
        )

        MAX_STYLED_ROWS = 10_000
        if len(display_df) <= MAX_STYLED_ROWS:
            def highlight_churn(row):
                if row["Churn_Label"] == "Yes":
                    return ["background-color: rgba(244,63,94,0.15)"] * len(row)
                return [""] * len(row)

            styled = display_df.style.apply(highlight_churn, axis=1).format(
                {"Churn_Probability": "{:.2%}"}
            )
            st.dataframe(styled, use_container_width=True, height=500)
        else:
            st.info(f"📋 Data terlalu besar untuk styling ({len(display_df):,} baris). Menampilkan tabel biasa.")
            st.dataframe(display_df, use_container_width=True, height=500)

        st.markdown("---")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "📥 Download CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv", mime="text/csv",
                use_container_width=True, key="dl_csv",
            )
        with dl2:
            st.download_button(
                "📥 Download JSON",
                data=result_df.to_json(orient="records", indent=2).encode("utf-8"),
                file_name="churn_predictions.json", mime="application/json",
                use_container_width=True, key="dl_json",
            )
        with dl3:
            st.download_button(
                "📥 Download Parquet",
                data=result_df.to_parquet(index=False),
                file_name="churn_predictions.parquet", mime="application/octet-stream",
                use_container_width=True, key="dl_parquet",
            )

    # visualization
    with tab_viz:
        v1, v2 = st.columns(2)
        with v1:
            st.markdown('<p class="dashboard-subheader">📌 Distribusi Prediksi</p>', unsafe_allow_html=True)
            pred_dist = result_df["Churn_Label"].value_counts().reset_index()
            pred_dist.columns = ["Churn", "Count"]
            fig_pie = px.pie(
                pred_dist, values="Count", names="Churn", hole=0.55,
                color="Churn", color_discrete_map=CHURN_COLORS,
            )
            fig_pie.update_traces(
                textinfo="percent+label", textfont_size=14,
                marker=dict(line=dict(color="#1a1a3e", width=2)),
            )
            fig_pie.update_layout(
                **PLOTLY_LAYOUT, showlegend=False, height=380,
                annotations=[dict(
                    text=f"<b>{churn_pct:.1f}%</b><br>Churn",
                    x=0.5, y=0.5, font_size=18, font_color="#f43f5e", showarrow=False,
                )],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with v2:
            st.markdown('<p class="dashboard-subheader">⚠️ Distribusi Risk Level</p>', unsafe_allow_html=True)
            risk_dist = (
                result_df["Risk_Level"].value_counts()
                .reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
            )
            risk_dist.columns = ["Risk", "Count"]
            fig_risk = px.bar(
                risk_dist, x="Risk", y="Count", color="Risk",
                color_discrete_map={"High": "#f43f5e", "Medium": "#fb923c", "Low": "#34d399"},
                text="Count",
            )
            fig_risk.update_traces(textposition="outside", textfont_size=13)
            fig_risk.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                                    xaxis_title="", yaxis_title="Jumlah")
            st.plotly_chart(fig_risk, use_container_width=True)

        v3, v4 = st.columns(2)
        with v3:
            st.markdown('<p class="dashboard-subheader">📈 Distribusi Probabilitas Churn</p>', unsafe_allow_html=True)
            fig_hist = px.histogram(
                result_df, x="Churn_Probability", nbins=40,
                color="Churn_Label", barmode="overlay",
                color_discrete_map=CHURN_COLORS, opacity=0.75,
            )
            fig_hist.add_vline(
                x=0.5, line_dash="dash", line_color="#fb923c",
                annotation_text="Threshold 0.5", annotation_font_color="#fb923c",
            )
            fig_hist.update_layout(**PLOTLY_LAYOUT, height=380,
                                    xaxis_title="Churn Probability", yaxis_title="Jumlah",
                                    xaxis_tickformat=".0%")
            st.plotly_chart(fig_hist, use_container_width=True)

        with v4:
            st.markdown('<p class="dashboard-subheader">🔝 Top 10 Pelanggan Berisiko Tinggi</p>', unsafe_allow_html=True)
            id_col = "id" if "id" in result_df.columns else "row_index"
            top10 = result_df.nlargest(10, "Churn_Probability")
            fig_top = px.bar(
                top10, y=top10[id_col].astype(str), x="Churn_Probability",
                orientation="h", color="Churn_Probability",
                color_continuous_scale=["#6366f1", "#f43f5e"],
                text=top10["Churn_Probability"].apply(lambda x: f"{x:.1%}"),
            )
            fig_top.update_traces(textposition="outside", textfont_size=11)
            fig_top.update_layout(
                **PLOTLY_LAYOUT, height=380,
                xaxis_title="Churn Probability", yaxis_title=id_col,
                xaxis_tickformat=".0%", coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_top, use_container_width=True)

        st.markdown('<p class="dashboard-subheader">📋 Prediksi Churn by Contract Type</p>', unsafe_allow_html=True)
        ct_dist = (
            result_df.groupby("Contract")["Churn_Label"]
            .value_counts(normalize=True).rename("Proportion").reset_index()
        )
        fig_ct = px.bar(
            ct_dist, x="Contract", y="Proportion", color="Churn_Label",
            barmode="group", color_discrete_map=CHURN_COLORS,
            text=ct_dist["Proportion"].apply(lambda x: f"{x:.1%}"),
        )
        fig_ct.update_traces(textposition="outside", textfont_size=11)
        fig_ct.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_tickformat=".0%",
                              xaxis_title="", yaxis_title="Proporsi")
        st.plotly_chart(fig_ct, use_container_width=True)
