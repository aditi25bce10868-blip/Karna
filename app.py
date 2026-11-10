import streamlit as st
import json
import os
import subprocess
import sys

# 1. Page Configuration & Full-Screen Layout
st.set_page_config(
    page_title="Karna: Haptic Assistive Wristband",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Advanced Professional CSS & Typography Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit Default UI Elements */
    header {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stDeployButton {display:none;}
    
    /* Global Application Font & Dark Theme */
    .stApp {
        background-color: #05070b;
        color: #f3f4f6;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 620px;
    }
    
    /* Login Card Styling */
    .login-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid #312e81;
        padding: 35px 25px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 15px 30px rgba(0,0,0,0.6);
        margin-top: 6vh;
    }
    
    /* Fix Input Box Visibility & Contrast */
    .stTextInput input {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }
    .stTextInput label {
        color: #93c5fd !important;
        font-weight: 500;
    }

    /* Metric & Visual Cards */
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Probability Badge Styling */
    .prob-badge {
        background-color: #0f172a;
        border: 1px solid #334155;
        padding: 8px 10px;
        border-radius: 8px;
        text-align: center;
        font-size: 0.8rem;
    }

    /* Safety Guidance Suggestion Box */
    .safety-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid #6366f1;
        padding: 14px 18px;
        border-radius: 10px;
        font-size: 0.9rem;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #e11d48;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 12px rgba(225, 29, 72, 0.3);
    }
    .stButton>button:hover {
        background-color: #be123c;
    }
    </style>
""", unsafe_allow_html=True)

AUDIO_FILE = os.path.join("scripts", "recorded.wav")
EVENT_FILE = os.path.join("pulse_events", "latest_event.json")

os.makedirs("scripts", exist_ok=True)
os.makedirs("pulse_events", exist_ok=True)

# 3. Session State Management
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- LOGIN SCREEN ---
if not st.session_state.logged_in:
    # Clear any lingering old data on login screen view
    if os.path.exists(EVENT_FILE):
        os.remove(EVENT_FILE)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="login-card">
            <h1 style="color: #38bdf8; font-size: 1.9rem; margin-bottom: 5px; font-weight:700;">🛡️ KARNA</h1>
            <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 20px;">Haptic Assistive Safety Wristband System</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2.2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        user_input = st.text_input("Operator Identification", placeholder="Enter operator name...")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Initialize System"):
            if user_input.strip():
                st.session_state.logged_in = True
                st.session_state.username = user_input.strip()
                st.rerun()
            else:
                st.warning("Please enter a valid operator name to proceed.")
    st.stop()

# --- MAIN DASHBOARD SCREEN (Post-Login) ---

# Header Bar
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown(f'<p style="font-size:1.2rem; font-weight:700; color:#38bdf8; margin:0;">🛡️ Karna Live Telemetry</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:0.75rem; color:#94a3b8; margin:0;">Operator: <b>{st.session_state.username}</b> | Status: Secure Link Active</p>', unsafe_allow_html=True)
with head_col2:
    if st.button("🔓 Lock Session"):
        st.session_state.logged_in = False
        if os.path.exists(EVENT_FILE):
            os.remove(EVENT_FILE)
        st.rerun()

st.markdown("<hr style='margin: 8px 0px; border-color: #1e293b;'>", unsafe_allow_html=True)

def execute_live_pipeline():
    rec_result = subprocess.run(
        [sys.executable, "scripts/record_wav.py", "--out", AUDIO_FILE, "--seconds", "12", "--device", "1"],
        capture_output=True, text=True
    )
    bridge_result = subprocess.run(
        [sys.executable, "pulse_bridge.py", "--audio", AUDIO_FILE],
        capture_output=True, text=True
    )
    return rec_result, bridge_result

# Action Control Button
if st.button("🔴 Record Live Audio & Classify Hazard"):
    if os.path.exists(EVENT_FILE):
        os.remove(EVENT_FILE)
    with st.spinner("🎙️ Capturing live audio & mapping haptic telemetry..."):
        execute_live_pipeline()

st.markdown("<br>", unsafe_allow_html=True)

# Safety Guidance Mapping Helper (Bold & Direct)
def get_safety_advice(class_name):
    advice_map = {
        "ambulance_siren": "🚨 <b>EMERGENCY VEHICLE:</b> Clear roadway immediately. Shift to right sidewalk and yield intersection priority.",
        "fire_truck_siren": "🚒 <b>HEAVY EMERGENCY VEHICLE:</b> High sound pressure detected. Step well back from the curb immediately.",
        "police_siren": "🚓 <b>LAW ENFORCEMENT ALERT:</b> Remain stationary and clear of active lanes. Watch for official directives.",
        "short_horn_ahead": "⚠️ <b>PROXIMITY WARNING:</b> Vehicle ahead. Verify forward path and maintain defensive footing.",
        "long_horn_ahead": "⚠️ <b>HEAVY HORN ALERT:</b> Incoming hazard detected. Step away from blind corners instantly.",
        "short_horn_alongside": "🚗 <b>SIDEWAY TRAFFIC:</b> Vehicle passing closely on flank. Maintain straight line stability.",
        "long_horn_alongside": "🚛 <b>HEAVY TRANSPORT:</b> Large vehicle passing alongside. Hold steady clear of wide turns.",
        "dog_bark_close": "🐕 <b>CANINE PROXIMITY:</b> Aggressive/nearby animal detected. Avoid sudden moves; back away slowly.",
        "dog_bark_far": "🐕 <b>DISTANT ANIMAL:</b> Canine vocalization in perimeter. Maintain situational awareness.",
        "shout_screaming": "🗣️ <b>DISTRESS SIGNAL:</b> Shouting/Scream identified. Scan immediate surroundings for assistance.",
        "construction_heavy": "🏗️ <b>HEAVY MACHINERY:</b> Construction zone noise logged. Wear high-visibility awareness.",
        "traffic_noise": "🏙️ <b>URBAN TRAFFIC:</b> Standard environmental ambient noise. Proceed with standard caution."
    }
    return advice_map.get(class_name, "🛡️ <b>SAFETY GUIDANCE:</b> Hazard detected. Exercise caution and scan your surroundings.")

# Results Presentation (Clean Graphical Single-Screen Grid)
if os.path.exists(EVENT_FILE):
    with open(EVENT_FILE, "r") as f:
        data = json.load(f)

    pred = data.get("prediction", {})
    pulse = data.get("pulse", {})
    className = pred.get("class_name", "Unknown")

    # Top Metrics Row
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-card"><small style="color:#94a3b8; font-size:0.75rem;">DETECTED THREAT</small><br><b style="font-size:0.95rem; color:#38bdf8;">{className}</b></div>', unsafe_allow_html=True)
    with c2:
        conf_val = pred.get("confidence", 0) * 100
        st.markdown(f'<div class="metric-card"><small style="color:#94a3b8; font-size:0.75rem;">CONFIDENCE LEVEL</small><br><b style="font-size:1.05rem; color:#4ade80;">{conf_val:.1f}%</b></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphical Top-3 Probability Mini-Badges Row
    st.markdown("<p style='font-size:0.75rem; font-weight:600; color:#94a3b8; margin-bottom:5px; text-transform:uppercase; letter-spacing:0.5px;'>📊 Top Confidence Breakdown</p>", unsafe_allow_html=True)
    top_3_items = pred.get("top_3", [])[:3]
    if top_3_items:
        p_cols = st.columns(len(top_3_items))
        for idx, item in enumerate(top_3_items):
            with p_cols[idx]:
                pct = item['confidence'] * 100
                st.markdown(f'<div class="prob-badge"><span style="color:#cbd5e1; font-weight:500;">{item["class_name"]}</span><br><b style="color:#38bdf8; font-size:0.9rem;">{pct:.1f}%</b></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dynamic Safety Guidance Suggestion Box (Bold & Prominent)
    advice_text = get_safety_advice(className)
    st.markdown(f'<div class="safety-box">{advice_text}</div>', unsafe_allow_html=True)

else:
    # Clean Initial Startup State
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👋 System ready. Click **'Record Live Audio & Classify Hazard'** above to test your live microphone input.")
    st.markdown("<br><br>", unsafe_allow_html=True)
