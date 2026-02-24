"""
1_📋_Prediction_Form.py
-----------------------
Streamlit Page 1: Student data input form and burnout risk prediction.
"""

import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from utils.helpers import get_model_path, model_exists, risk_color, risk_emoji, format_probability
from src.inference import load_model, predict, build_input_dict

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Burnout Prediction Form",
    page_icon="clipboard",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .page-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f3460 100%);
        border-radius: 12px;
        padding: 32px 36px;
        margin-bottom: 28px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .page-header h2 { color: white; margin: 0 0 6px 0; font-size: 1.8rem; }
    .page-header p  { color: rgba(255,255,255,0.65); margin: 0; }

    .result-card {
        border-radius: 16px;
        padding: 36px;
        text-align: center;
        margin-top: 24px;
        border: 2px solid;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);} }
    .result-card .risk-label { font-size: 3rem; font-weight: 700; }
    .result-card .risk-sub   { font-size: 1.1rem; margin-top: 8px; opacity: 0.8; }

    .prob-bar-label { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 4px; }

    [data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid rgba(255,255,255,0.06); }
    footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h2>Burnout Risk Assessment Form</h2>
    <p>Fill in your current academic and wellbeing details to receive a personalised burnout risk prediction.</p>
</div>
""", unsafe_allow_html=True)

# ── Check model ───────────────────────────────────────────────────────────────
if not model_exists():
    st.error(
        "**Model not found.** Please run `python train.py` from the project root first.",
        icon="🚨"
    )
    st.stop()

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("prediction_form", clear_on_submit=False):
    st.markdown("#### Personal & Academic Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
        age = st.number_input("Age", min_value=17, max_value=40, value=21, step=1)
        cgpa = st.number_input("CGPA (out of 10.0)", min_value=0.0, max_value=10.0, value=8.0, step=0.1, format="%.1f")

    with col2:
        course = st.text_input("Course / Major", value="Engineering", placeholder="e.g. Engineering")
        year_map = {"Year 1": "Year 1", "Year 2": "Year 2", "Year 3": "Year 3", "Year 4": "Year 4"}
        year_of_study = st.selectbox("Year of Study", options=list(year_map.keys()), index=1)
        study_hours = st.slider("Study Hours per Week", min_value=1, max_value=40, value=18)

    with col3:
        academic_engagement = st.slider("Academic Engagement (1=Low, 5=High)", 1, 5, 3)
        specialist_treatment = st.radio(
            "Currently receiving specialist treatment?",
            options=["No", "Yes"], horizontal=True
        )
        has_support = st.radio(
            "Have mental health support access?",
            options=["No", "Yes"], horizontal=True
        )

    st.divider()
    st.markdown("#### Mental Health Symptoms")
    col4, col5 = st.columns(2)

    with col4:
        depression = st.radio("Experiencing Depression?", options=["No", "Yes"], horizontal=True)
        anxiety = st.radio("Experiencing Anxiety?", options=["No", "Yes"], horizontal=True)
        panic_attack = st.radio("Experiencing Panic Attacks?", options=["No", "Yes"], horizontal=True)
        symptom_freq = st.slider(
            "Symptom Days in Last 7 Days", min_value=0, max_value=7, value=3,
            help="How many of the last 7 days did you experience mental health symptoms?"
        )

    with col5:
        sleep_quality = st.slider(
            "Sleep Quality (1=Very Poor, 5=Excellent)", min_value=1, max_value=5, value=3
        )
        study_stress = st.slider(
            "Study Stress Level (1=Very Low, 5=Very High)", min_value=1, max_value=5, value=3
        )

    submitted = st.form_submit_button("Predict My Burnout Risk", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # Convert radio button values
    def yn_to_int(val): return 1 if val == "Yes" else 0

    input_dict = build_input_dict(
        gender=gender,
        age=int(age),
        course=course.strip().title(),
        year_of_study=year_of_study,
        cgpa=float(cgpa) / 10.0 * 4.0,  
        depression=yn_to_int(depression),
        anxiety=yn_to_int(anxiety),
        panic_attack=yn_to_int(panic_attack),
        specialist_treatment=yn_to_int(specialist_treatment),
        symptom_frequency=int(symptom_freq),
        has_mental_health_support=yn_to_int(has_support),
        sleep_quality=int(sleep_quality),
        study_stress_level=int(study_stress),
        study_hours_per_week=int(study_hours),
        academic_engagement=int(academic_engagement),
    )

    with st.spinner("Analysing your inputs..."):
        bundle = load_model(get_model_path())
        risk_label, prob_dict, risk_score = predict(bundle, input_dict)

    # Save to session state for chatbot and dashboard
    st.session_state["risk_label"] = risk_label
    st.session_state["prob_dict"] = prob_dict
    st.session_state["risk_score"] = risk_score
    st.session_state["last_input"] = input_dict

    # ── Result Card ───────────────────────────────────────────────────────────
    color = risk_color(risk_label)
    emoji = risk_emoji(risk_label)

    st.markdown(f"""
    <div class="result-card" style="background: {color}18; border-color: {color};">
        <div class="risk-label" style="color:{color};">{emoji} {risk_label} Risk</div>
        <div class="risk-sub" style="color:{color};">
            Confidence: {format_probability(risk_score)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability Breakdown ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Probability Breakdown")
    cols = st.columns(len(prob_dict))
    risk_order = ["Low", "Medium", "High"]
    ordered = [(k, prob_dict[k]) for k in risk_order if k in prob_dict]

    for i, (label, prob) in enumerate(ordered):
        with cols[i]:
            c = risk_color(label)
            st.markdown(f"<div class='prob-bar-label'>{risk_emoji(label)} {label}</div>", unsafe_allow_html=True)
            st.progress(prob)
            st.markdown(f"<center style='color:{c}; font-weight:600;'>{format_probability(prob)}</center>", unsafe_allow_html=True)

    # ── Actionable Message ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if risk_label == "High":
        st.error(
            "**High burnout risk detected.** Consider visiting the **AI Chatbot** page for immediate stress relief exercises, "
            "and speak with a counsellor or mental health professional.",
            icon="🚨"
        )
    elif risk_label == "Medium":
        st.warning(
            "**Medium burnout risk detected.** You're showing some signs of stress and fatigue. "
            "Visit the **AI Chatbot** page for helpful exercises to maintain your balance.",
            icon="⚠️"
        )
    else:
        st.success(
            "**Low burnout risk.** You appear to be managing well. Keep up good habits! "
            "Visit the **AI Chatbot** for tips to maintain your wellbeing.",
            icon="✅"
        )

    st.markdown(
        "_Navigate to **Results Dashboard** to see model performance metrics, "
        "or **AI Chatbot** for personalised exercise recommendations._"
    )
