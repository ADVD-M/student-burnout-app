"""
2_📊_Results_Dashboard.py
--------------------------
Streamlit Page 2: Model evaluation metrics, confusion matrix,
and ROC AUC visualization.
"""

import sys
import pathlib
import json

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from utils.helpers import get_model_path, model_exists, sort_feature_importances, clean_feature_name

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Results Dashboard",
    page_icon="bar_chart",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .page-header {
        background: linear-gradient(135deg, #0c2340 0%, #163270 100%);
        border-radius: 12px;
        padding: 32px 36px;
        margin-bottom: 28px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .page-header h2 { color: white; margin: 0 0 6px 0; font-size: 1.8rem; }
    .page-header p  { color: rgba(255,255,255,0.65); margin: 0; }

    .metric-card {
        background: #1e2535;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 22px 20px;
        text-align: center;
    }
    .metric-card .metric-value { font-size: 2.2rem; font-weight: 700; color: #60a5fa; }
    .metric-card .metric-label { font-size: 0.82rem; color: rgba(255,255,255,0.5); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

    .section-title { font-size: 1.15rem; font-weight: 600; color: #e2e8f0; margin: 24px 0 12px 0; }

    [data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid rgba(255,255,255,0.06); }
    footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h2>Model Results Dashboard</h2>
    <p>Comprehensive evaluation metrics, confusion matrix, and per-class performance for the trained burnout risk model.</p>
</div>
""", unsafe_allow_html=True)

# ── Load metrics ──────────────────────────────────────────────────────────────
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

if not model_exists():
    st.error("**Model not found.** Please run `python train.py` first.", icon="🚨")
    st.stop()

if not METRICS_PATH.exists():
    st.warning(
        "Metrics file not found. Run `python train.py` to generate metrics.",
        icon="⚠️"
    )
    st.stop()

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

# ── Metric Cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Core Evaluation Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
metric_items = [
    ("Accuracy",  f"{metrics['accuracy']:.1%}"),
    ("Precision", f"{metrics['precision']:.1%}"),
    ("Recall",    f"{metrics['recall']:.1%}"),
    ("F1 Score",  f"{metrics['f1_score']:.1%}"),
    ("ROC AUC",   f"{metrics['roc_auc']:.3f}" if metrics.get("roc_auc") else "N/A"),
]

for col, (label, value) in zip([col1, col2, col3, col4, col5], metric_items):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Confusion Matrix ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

    class_names = metrics.get("class_names", ["High", "Low", "Medium"])
    cm = np.array(metrics["confusion_matrix"])

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred: {c}" for c in class_names],
        y=[f"True: {c}" for c in class_names],
        colorscale=[
            [0.0, "#0f172a"],
            [0.5, "#1e40af"],
            [1.0, "#3b82f6"],
        ],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 18, "color": "white"},
        showscale=False,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=360,
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ── Per-class Metrics Table ───────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-title">Per-Class Performance</div>', unsafe_allow_html=True)

    report = metrics.get("classification_report", {})
    table_data = []
    for cls in class_names:
        if cls in report:
            r = report[cls]
            table_data.append({
                "Class": cls,
                "Precision": f"{r['precision']:.3f}",
                "Recall": f"{r['recall']:.3f}",
                "F1-Score": f"{r['f1-score']:.3f}",
                "Support": int(r["support"]),
            })

    if table_data:
        df_report = pd.DataFrame(table_data)
        st.dataframe(
            df_report.set_index("Class"),
            use_container_width=True,
            height=200,
        )

    # ── Prediction from session ───────────────────────────────────────────────
    if "risk_label" in st.session_state:
        risk = st.session_state["risk_label"]
        prob = st.session_state["prob_dict"]
        from utils.helpers import risk_color, risk_emoji
        color = risk_color(risk)
        st.info(
            f"**Last Prediction:** {risk_emoji(risk)} **{risk}** risk  "
            f"({', '.join([f'{k}: {v*100:.1f}%' for k, v in prob.items()])})",
            icon="📌"
        )

st.divider()

# ── Probability Distribution ──────────────────────────────────────────────────
if "prob_dict" in st.session_state:
    st.markdown('<div class="section-title">Your Risk Probability Distribution</div>', unsafe_allow_html=True)

    prob_dict = st.session_state["prob_dict"]
    risk_order = ["Low", "Medium", "High"]
    colors_map = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}

    ordered = [(k, prob_dict.get(k, 0)) for k in risk_order]
    fig_prob = go.Figure(go.Bar(
        x=[k for k, _ in ordered],
        y=[v for _, v in ordered],
        marker_color=[colors_map[k] for k, _ in ordered],
        text=[f"{v*100:.1f}%" for _, v in ordered],
        textposition="outside",
        textfont=dict(color="white", size=14),
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig_prob.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        yaxis=dict(
            tickformat=".0%",
            range=[0, 1],
            gridcolor="rgba(255,255,255,0.06)",
        ),
        xaxis=dict(title="Risk Category"),
        showlegend=False,
    )
    st.plotly_chart(fig_prob, use_container_width=True)
