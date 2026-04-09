"""
AI Agent module for Rini — AI Strategic Business Consultant.
Uses OpenAI Function Calling to provide:
  - RAG knowledge search (ChromaDB)
  - Churn prediction (XGBoost model)
  - Data analysis (pandas)
  - Chart generation (plotly)
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- OpenAI client ----------
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- lazy-loaded resources ----------
_df_cache = None
_rag_ready = False


def _get_dataframe() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv("data/train.csv").drop(columns=["id"], errors="ignore")
    return _df_cache


def _ensure_rag():
    """Import rag_chatbot lazily to avoid ChromaDB boot at import time."""
    global _rag_ready
    if not _rag_ready:
        from llm import rag_chatbot as _rc  # noqa: F401
        _rag_ready = True


# ──────────────────────────────────────────────
# TOOL DEFINITIONS (OpenAI function-calling schema)
# ──────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the Nusantara Connect knowledge base (company profile, "
                "policies, services, churn strategy) using RAG. "
                "Use this when the user asks about company info, services, "
                "policies, or business context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query in Indonesian or English.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_churn",
            "description": (
                "Predict whether a single customer will churn using the trained "
                "XGBoost model. Returns prediction label, probability, and risk "
                "level. Use when the user asks to predict churn for a customer "
                "with specific attributes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gender": {"type": "string", "enum": ["Male", "Female"]},
                    "SeniorCitizen": {"type": "integer", "enum": [0, 1], "description": "0=No, 1=Yes"},
                    "Partner": {"type": "string", "enum": ["Yes", "No"]},
                    "Dependents": {"type": "string", "enum": ["Yes", "No"]},
                    "tenure": {"type": "integer", "description": "Number of months the customer has stayed (0-72)"},
                    "PhoneService": {"type": "string", "enum": ["Yes", "No"]},
                    "MultipleLines": {"type": "string", "enum": ["Yes", "No", "No phone service"]},
                    "InternetService": {"type": "string", "enum": ["DSL", "Fiber optic", "No"]},
                    "OnlineSecurity": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
                    "OnlineBackup": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
                    "DeviceProtection": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
                    "TechSupport": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
                    "StreamingTV": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
                    "StreamingMovies": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
                    "Contract": {"type": "string", "enum": ["Month-to-month", "One year", "Two year"]},
                    "PaperlessBilling": {"type": "string", "enum": ["Yes", "No"]},
                    "PaymentMethod": {
                        "type": "string",
                        "enum": [
                            "Electronic check", "Mailed check",
                            "Credit card (automatic)", "Bank transfer (automatic)",
                        ],
                    },
                    "MonthlyCharges": {"type": "number", "description": "Monthly charges in dollars"},
                    "TotalCharges": {"type": "number", "description": "Total charges in dollars"},
                },
                "required": [
                    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                    "PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod",
                    "MonthlyCharges", "TotalCharges",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": (
                "Analyze the customer churn dataset using pandas. "
                "Supports: churn_rate_by (group by a column and get churn rate), "
                "distribution (value counts of a column), "
                "statistics (descriptive stats of a numeric column), "
                "correlation (correlation between numeric columns), "
                "cross_tab (cross tabulation of two columns with churn). "
                "Use when user asks analytical questions about the data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "enum": ["churn_rate_by", "distribution", "statistics", "correlation", "cross_tab"],
                        "description": "Type of analysis to perform.",
                    },
                    "column": {
                        "type": "string",
                        "description": (
                            "Primary column to analyze. Available columns: "
                            "gender, SeniorCitizen, Partner, Dependents, tenure, "
                            "PhoneService, MultipleLines, InternetService, "
                            "OnlineSecurity, OnlineBackup, DeviceProtection, "
                            "TechSupport, StreamingTV, StreamingMovies, "
                            "Contract, PaperlessBilling, PaymentMethod, "
                            "MonthlyCharges, TotalCharges, Churn"
                        ),
                    },
                    "column2": {
                        "type": "string",
                        "description": "Secondary column (for cross_tab analysis).",
                    },
                    "filter_column": {
                        "type": "string",
                        "description": "Optional: column to filter on before analysis.",
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "Optional: value to filter the filter_column by.",
                    },
                },
                "required": ["analysis_type", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": (
                "Create a plotly chart/visualization from the customer data. "
                "The chart will be displayed directly in the chat. "
                "Use when user asks for a visualization, graph, or chart."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "pie", "histogram", "box", "scatter", "heatmap"],
                        "description": "Type of chart to create.",
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Column for x-axis.",
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Column for y-axis (optional for some chart types).",
                    },
                    "color_by": {
                        "type": "string",
                        "description": "Column to color/group by (e.g. 'Churn').",
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title.",
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": ["count", "mean", "sum", "churn_rate"],
                        "description": "How to aggregate data for bar charts.",
                    },
                    "filter_column": {
                        "type": "string",
                        "description": "Optional: column to filter before charting.",
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "Optional: value to filter by.",
                    },
                },
                "required": ["chart_type", "title"],
            },
        },
    },
]


# ──────────────────────────────────────────────
# TOOL EXECUTORS
# ──────────────────────────────────────────────

CHURN_COLORS = {"Yes": "#f43f5e", "No": "#6366f1", 1: "#f43f5e", 0: "#6366f1"}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
)


def _exec_search_knowledge(args: dict) -> str:
    _ensure_rag()
    from llm.rag_chatbot import search
    results = search(args["query"], n_result=3)
    if not results:
        return (
            "Tidak ditemukan informasi relevan dalam knowledge base. "
            "Pertanyaan ini kemungkinan di luar cakupan domain Nusantara Connect. "
            "Tolak dengan sopan jika pertanyaan tidak terkait dengan "
            "perusahaan, layanan, atau analisis pelanggan."
        )
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] (source: {r['source']}, relevance: {1 - r['distance']:.2f})\n{r['text']}")
    return "\n\n".join(parts)


def _exec_predict_churn(args: dict) -> str:
    import batch_predict

    row = pd.DataFrame([{
        "gender": args["gender"],
        "SeniorCitizen": args["SeniorCitizen"],
        "Partner": args["Partner"],
        "Dependents": args["Dependents"],
        "tenure": args["tenure"],
        "PhoneService": args["PhoneService"],
        "MultipleLines": args["MultipleLines"],
        "InternetService": args["InternetService"],
        "OnlineSecurity": args["OnlineSecurity"],
        "OnlineBackup": args["OnlineBackup"],
        "DeviceProtection": args["DeviceProtection"],
        "TechSupport": args["TechSupport"],
        "StreamingTV": args["StreamingTV"],
        "StreamingMovies": args["StreamingMovies"],
        "Contract": args["Contract"],
        "PaperlessBilling": args["PaperlessBilling"],
        "PaymentMethod": args["PaymentMethod"],
        "MonthlyCharges": args["MonthlyCharges"],
        "TotalCharges": args["TotalCharges"],
    }])

    try:
        predictions, probabilities = batch_predict._batch_inference(row)
        pred = int(predictions[0])
        proba = float(probabilities[0])
        risk = "High" if proba >= 0.8 else "Medium" if proba >= 0.5 else "Low"
        label = "CHURN" if pred == 1 else "STAY"

        return json.dumps({
            "prediction": label,
            "churn_probability": round(proba * 100, 2),
            "risk_level": risk,
            "input_summary": {
                "gender": args["gender"],
                "tenure": args["tenure"],
                "InternetService": args["InternetService"],
                "Contract": args["Contract"],
                "MonthlyCharges": args["MonthlyCharges"],
                "PaymentMethod": args["PaymentMethod"],
            },
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _exec_analyze_data(args: dict) -> str:
    df = _get_dataframe()
    analysis = args["analysis_type"]
    col = args["column"]

    # optional filter
    if args.get("filter_column") and args.get("filter_value"):
        fc, fv = args["filter_column"], args["filter_value"]
        if fc in df.columns:
            df = df[df[fc].astype(str) == str(fv)]

    if col not in df.columns:
        return json.dumps({"error": f"Kolom '{col}' tidak ditemukan. Kolom tersedia: {list(df.columns)}"})

    try:
        if analysis == "churn_rate_by":
            result = (
                df.groupby(col)["Churn"]
                .apply(lambda x: (x == "Yes").mean())
                .sort_values(ascending=False)
            )
            out = {str(k): f"{v:.2%}" for k, v in result.items()}
            total_churn = (df["Churn"] == "Yes").mean()
            return json.dumps({
                "analysis": f"Churn Rate by {col}",
                "total_records": len(df),
                "overall_churn_rate": f"{total_churn:.2%}",
                "results": out,
            }, ensure_ascii=False)

        elif analysis == "distribution":
            result = df[col].value_counts()
            out = {str(k): int(v) for k, v in result.items()}
            return json.dumps({
                "analysis": f"Distribution of {col}",
                "total_records": len(df),
                "results": out,
            }, ensure_ascii=False)

        elif analysis == "statistics":
            if not pd.api.types.is_numeric_dtype(df[col]):
                return json.dumps({"error": f"Kolom '{col}' bukan kolom numerik."})
            stats = df[col].describe()
            out = {str(k): round(float(v), 2) for k, v in stats.items()}
            # add stats by churn
            churn_stats = df.groupby("Churn")[col].mean()
            out["mean_churn_yes"] = round(float(churn_stats.get("Yes", 0)), 2)
            out["mean_churn_no"] = round(float(churn_stats.get("No", 0)), 2)
            return json.dumps({
                "analysis": f"Statistics of {col}",
                "results": out,
            }, ensure_ascii=False)

        elif analysis == "correlation":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if col not in numeric_cols:
                return json.dumps({"error": f"Kolom '{col}' bukan numerik."})
            corr = df[numeric_cols].corr()[col].sort_values(ascending=False)
            out = {str(k): round(float(v), 4) for k, v in corr.items()}
            return json.dumps({
                "analysis": f"Correlation with {col}",
                "results": out,
            }, ensure_ascii=False)

        elif analysis == "cross_tab":
            col2 = args.get("column2", "Churn")
            if col2 not in df.columns:
                return json.dumps({"error": f"Kolom '{col2}' tidak ditemukan."})
            ct = pd.crosstab(df[col], df[col2], normalize="index")
            rows = {}
            for idx in ct.index:
                rows[str(idx)] = {str(c): f"{ct.loc[idx, c]:.2%}" for c in ct.columns}
            return json.dumps({
                "analysis": f"Cross-tab: {col} vs {col2}",
                "results": rows,
            }, ensure_ascii=False)

        else:
            return json.dumps({"error": f"Unknown analysis type: {analysis}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


def _exec_create_chart(args: dict) -> dict:
    """Returns a dict with 'figure' (plotly.Figure) and 'summary' (str)."""
    df = _get_dataframe()
    chart_type = args["chart_type"]
    title = args.get("title", "Chart")
    x = args.get("x_column")
    y = args.get("y_column")
    color = args.get("color_by")
    agg = args.get("aggregation", "count")

    # optional filter
    if args.get("filter_column") and args.get("filter_value"):
        fc, fv = args["filter_column"], args["filter_value"]
        if fc in df.columns:
            df = df[df[fc].astype(str) == str(fv)]

    color_map = CHURN_COLORS if color == "Churn" else None

    try:
        fig = None
        summary = ""

        if chart_type == "bar":
            if agg == "churn_rate" and x:
                agg_df = (
                    df.groupby(x)["Churn"]
                    .apply(lambda s: (s == "Yes").mean())
                    .sort_values(ascending=False)
                    .reset_index()
                )
                agg_df.columns = [x, "Churn_Rate"]
                fig = px.bar(
                    agg_df, x=x, y="Churn_Rate", title=title,
                    text=agg_df["Churn_Rate"].apply(lambda v: f"{v:.1%}"),
                    color="Churn_Rate",
                    color_continuous_scale=["#6366f1", "#f43f5e"],
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
                summary = f"Bar chart: Churn Rate by {x}"
            elif agg == "mean" and x and y:
                agg_df = df.groupby(x)[y].mean().reset_index()
                fig = px.bar(agg_df, x=x, y=y, title=title,
                             text=agg_df[y].apply(lambda v: f"{v:.1f}"))
                fig.update_traces(textposition="outside")
                summary = f"Bar chart: Mean {y} by {x}"
            elif x:
                if color:
                    fig = px.histogram(
                        df, x=x, color=color, barmode="group",
                        title=title, color_discrete_map=color_map,
                    )
                else:
                    fig = px.histogram(df, x=x, title=title)
                summary = f"Bar chart: Count of {x}"

        elif chart_type == "pie":
            col_pie = x or color or "Churn"
            dist = df[col_pie].value_counts().reset_index()
            dist.columns = [col_pie, "Count"]
            fig = px.pie(
                dist, values="Count", names=col_pie, title=title,
                hole=0.5, color=col_pie, color_discrete_map=color_map,
            )
            fig.update_traces(textinfo="percent+label")
            summary = f"Pie chart: Distribution of {col_pie}"

        elif chart_type == "histogram":
            col_hist = x or y or "MonthlyCharges"
            fig = px.histogram(
                df, x=col_hist, color=color, title=title,
                nbins=30, barmode="overlay", opacity=0.75,
                color_discrete_map=color_map,
            )
            summary = f"Histogram: {col_hist}"

        elif chart_type == "box":
            fig = px.box(
                df, x=x, y=y, color=color, title=title,
                color_discrete_map=color_map,
            )
            summary = f"Box plot: {y} by {x}"

        elif chart_type == "scatter":
            sample = df.sample(n=min(5000, len(df)), random_state=42)
            fig = px.scatter(
                sample, x=x, y=y, color=color, title=title,
                opacity=0.6, color_discrete_map=color_map,
            )
            summary = f"Scatter: {x} vs {y}"

        elif chart_type == "heatmap":
            if x and color:
                ct = pd.crosstab(df[x], df[color], normalize="index")
                fig = go.Figure(data=go.Heatmap(
                    z=ct.values, x=ct.columns.tolist(), y=ct.index.tolist(),
                    colorscale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#f43f5e"]],
                    text=[[f"{v:.1%}" for v in row] for row in ct.values],
                    texttemplate="%{text}", textfont=dict(size=13, color="white"),
                ))
                fig.update_layout(title=title)
                summary = f"Heatmap: {x} vs {color}"
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                corr = df[numeric_cols].corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#f43f5e"]],
                    text=[[f"{v:.2f}" for v in row] for row in corr.values],
                    texttemplate="%{text}", textfont=dict(size=12, color="white"),
                ))
                fig.update_layout(title=title)
                summary = "Heatmap: Correlation matrix"

        if fig is None:
            return {"figure": None, "summary": "Could not create chart with given parameters."}

        fig.update_layout(**PLOTLY_LAYOUT, height=420)
        return {"figure": fig, "summary": summary}

    except Exception as e:
        return {"figure": None, "summary": f"Error creating chart: {e}"}


# ──────────────────────────────────────────────
# TOOL DISPATCHER
# ──────────────────────────────────────────────

_EXECUTORS = {
    "search_knowledge": _exec_search_knowledge,
    "predict_churn": _exec_predict_churn,
    "analyze_data": _exec_analyze_data,
    "create_chart": _exec_create_chart,
}


# ──────────────────────────────────────────────
# AGENT LOOP
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are Rini, an AI Strategic Business Consultant and data strategy expert at PT Nusantara Komunikasi Terpadu (Nusantara Connect).

You have access to powerful tools:
1. **search_knowledge** — Search the company knowledge base (RAG) for company profile, services, policies, and business context.
2. **predict_churn** — Run the ML model (XGBoost) to predict whether a customer will churn. You need customer attributes to use this.
3. **analyze_data** — Analyze the customer dataset (594K records) with pandas. Get churn rates, distributions, statistics, correlations, and cross-tabs.
4. **create_chart** — Generate interactive charts (bar, pie, histogram, box, scatter, heatmap) from the data.

=== TOPIC BOUNDARY (STRICT RULE) ===
You may ONLY answer questions related to the following topics:
- Nusantara Connect (company profile, services, products, policies)
- Customer churn analysis, prediction, and retention strategy
- Customer data analysis and visualization from the dataset
- Telecommunications business strategy and operations
- Data-driven business recommendations for Nusantara Connect

If the user asks a question that is OUTSIDE these topics (e.g., general knowledge, coding help, math homework, recipes, politics, health advice, sports, entertainment, etc.), you MUST:
1. Politely decline to answer.
2. Explain that your expertise is specifically in Nusantara Connect business intelligence and customer churn analytics.
3. Suggest the user ask a relevant question instead.

Example rejection response:
"Mohon maaf, saya adalah AI Business Consultant khusus untuk Nusantara Connect. Saya hanya bisa membantu pertanyaan seputar analisis pelanggan, prediksi churn, layanan perusahaan, dan strategi bisnis Nusantara Connect. Silakan ajukan pertanyaan yang berkaitan dengan topik tersebut. 😊"

=== RESPONSE STYLE (FLEXIBLE & CONTEXTUAL) ===
- Answer based on actual DATA and CONTEXT, not by rigidly quoting regulations or policies.
- When the data provides a clear answer, lead with the data insight first.
- Only reference company policies/regulations when directly relevant to the user's question.
- Provide practical, actionable answers that are immediately useful for decision-making.
- Avoid generic or overly formal regulatory language unless specifically asked about policies.
- If the knowledge base provides relevant context, synthesize it into a natural, conversational answer rather than citing it verbatim.

Guidelines:
- Use tools proactively when the user asks questions that can be answered with data.
- When predicting churn, if the user doesn't provide all fields, use reasonable defaults (No for optional services, Male, 0 for SeniorCitizen, etc.) and mention which defaults you used.
- Always provide actionable business recommendations alongside data insights.
- Structure responses clearly with Key Insight, Business Implication, and Recommended Action when appropriate.
- When creating charts, choose the most appropriate chart type for the data.

Language Rule:
- Always respond in the same language used by the user.
- If the user uses Indonesian, respond fully in Indonesian.
- If the user uses English, respond fully in English.

Communication Style:
- Be direct, clear, and decision-oriented.
- Use emoji to make responses more engaging.
- Format data results in clean tables or bullet points.
"""


def run_agent(user_message: str, history: list) -> tuple[str, list]:
    """
    Run the agent loop.

    Args:
        user_message: The user's input message.
        history: List of message dicts (role/content) from prior turns.

    Returns:
        (response_text, charts)
        - response_text: The final text response from the agent.
        - charts: List of plotly Figure objects generated during the turn.
    """
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    charts = []
    max_iterations = 6  # safety: prevent infinite loops

    for _ in range(max_iterations):
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]

        # If the model wants to call tools
        if choice.finish_reason == "tool_calls" or choice.message.tool_calls:
            # Add assistant message with tool calls
            messages.append(choice.message.model_dump())

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                executor = _EXECUTORS.get(fn_name)
                if executor is None:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})
                else:
                    result = executor(fn_args)

                # Special handling for create_chart (returns dict with figure)
                if fn_name == "create_chart" and isinstance(result, dict):
                    if result.get("figure") is not None:
                        charts.append(result["figure"])
                    result = result.get("summary", "Chart created.")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

            # Continue the loop to let the model process tool results
            continue

        # Model returned a text response — we're done
        return choice.message.content or "", charts

    # If we hit max iterations, return what we have
    return "Maaf, proses terlalu kompleks. Silakan coba pertanyaan yang lebih sederhana.", charts
