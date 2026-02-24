"""
Home.py
-------
Main entry point for the Student Burnout Risk Prediction Streamlit app.
Run with:  streamlit run app/Home.py
"""

import sys
import pathlib

# ── Path bootstrap (works locally + Streamlit Cloud) ─────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from utils.helpers import model_exists

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Burnout Risk Predictor",
    page_icon="graduation_cap",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Prevent main content area from scrolling */
    [data-testid="stAppViewContainer"] > .main {
        overflow: hidden;
    }
    section.main > div {
        overflow: hidden;
    }

    /* Hero gradient header */
    .hero-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
        border-radius: 16px;
        padding: 48px 40px;
        text-align: center;
        margin-bottom: 32px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-box h1 {
        color: #ffffff;
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 12px;
        letter-spacing: -0.5px;
    }
    .hero-box p {
        color: rgba(255,255,255,0.72);
        font-size: 1.1rem;
        line-height: 1.7;
        max-width: 680px;
        margin: 0 auto;
    }

    /* Status banner */
    .status-ok {
        background: linear-gradient(90deg, #064e3b, #065f46);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 14px 20px;
        color: #a7f3d0;
        font-size: 0.95rem;
    }
    .status-warn {
        background: linear-gradient(90deg, #451a03, #78350f);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 14px 20px;
        color: #fde68a;
        font-size: 0.95rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    /* Hide default streamlit footer */
    footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <h1>Student Burnout Risk Predictor</h1>
    <p>
        An AI-powered tool that assesses burnout risk using academic and mental health indicators.
        Get personalised mental wellness recommendations powered by evidence-based techniques.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Model Status ──────────────────────────────────────────────────────────────
if model_exists():
    st.markdown("""
    <div class="status-ok">
        <strong>Model Ready</strong> — The trained model is loaded and ready for predictions.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-warn">
        <strong>Model Not Found</strong> — Please run <code>python train.py</code> from the project root to train and save the model before making predictions.
    </div>
    """, unsafe_allow_html=True)
