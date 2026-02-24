"""
3_🤖_AI_Chatbot.py
-------------------
Streamlit Page 3: AI mental health chatbot with exercise recommendations.
Picks up the user's burnout risk level from session state (set on Page 1).
"""

import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from chatbot.chatbot import get_response, get_welcome_message
from utils.helpers import risk_color, risk_emoji

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Wellness Companion",
    page_icon="robot_face",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .page-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 50%, #0d1b2a 100%);
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .page-header h2 { color: white; margin: 0 0 6px 0; font-size: 1.8rem; }
    .page-header p  { color: rgba(255,255,255,0.65); margin: 0; }

    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 600;
        letter-spacing: 0.4px;
    }

    .topic-chip {
        display: inline-block;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 999px;
        padding: 5px 14px;
        font-size: 0.82rem;
        margin: 3px;
        cursor: pointer;
        color: #cbd5e1;
        transition: background 0.2s;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    [data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid rgba(255,255,255,0.06); }
    footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h2>AI Wellness Companion</h2>
    <p>Ask me about breathing exercises, focus techniques, journaling, stress relief, or sleep hygiene.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — Risk Context & Quick Topics ────────────────────────────────────
with st.sidebar:
    st.markdown("### Your Risk Context")

    risk_level = st.session_state.get("risk_label", "Unknown")
    prob_dict = st.session_state.get("prob_dict", {})

    if risk_level != "Unknown":
        color = risk_color(risk_level)
        st.markdown(
            f'<span class="risk-badge" style="background:{color}22; color:{color}; border:1px solid {color};">'
            f'{risk_emoji(risk_level)} {risk_level} Burnout Risk</span>',
            unsafe_allow_html=True
        )
        if prob_dict:
            st.markdown("**Probabilities:**")
            for cls in ["Low", "Medium", "High"]:
                if cls in prob_dict:
                    st.progress(prob_dict[cls], text=f"{cls}: {prob_dict[cls]*100:.1f}%")
    else:
        st.info("Complete the **Prediction Form** (Page 1) first to get personalised responses.", icon="ℹ️")

    st.divider()
    st.markdown("### Quick Topics")
    st.caption("Type any of these or your own message:")
    topics = [
        "Help me breathe better",
        "I can't focus on studying",
        "I feel overwhelmed",
        "I can't sleep at night",
        "Give me a journaling prompt",
        "I'm stressed about exams",
    ]
    for topic in topics:
        st.markdown(f"- *{topic}*")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

# ── Chat History initialisation ───────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ── Add welcome message on first load ────────────────────────────────────────
if len(st.session_state["chat_history"]) == 0:
    welcome = get_welcome_message(risk_level)
    st.session_state["chat_history"].append({"role": "assistant", "content": welcome})

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state["chat_history"]:
    avatar = "🤖" if msg["role"] == "assistant" else "🧑‍🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── User Input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me about stress, breathing, focus, sleep, or journaling...")

if user_input:
    # Append user message
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = get_response(user_input, risk_level=risk_level)
        st.markdown(response)

    # Append assistant response
    st.session_state["chat_history"].append({"role": "assistant", "content": response})
