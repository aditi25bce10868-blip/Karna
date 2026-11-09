import streamlit as st
import json
import os

st.set_page_config(page_title="Karna Wristband Dashboard", page_icon="⚡", layout="centered")

st.title("🛡️ Karna: Haptic Assistive Wristband")
st.markdown("Real-time Emergency Sound Classification & Haptic Feedback Pipeline")

# Button to trigger the simulation or load latest event
if st.button("🔄 Check Latest Detection Event"):
    event_path = "pulse_events/latest_event.json"
    
    if os.path.exists(event_path):
        with open(event_path, "r") as f:
            data = json.load(f)
            
        prediction = data.get("prediction", {})
        pulse = data.get("pulse", {})
        
        st.success("Event Loaded Successfully!")
        
        # Display Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Detected Sound Class", value=prediction.get("class_name", "Unknown"))
        with col2:
            st.metric(label="Confidence Score", value=f"{prediction.get('confidence', 0)*100:.2f}%")
            
        st.markdown("### 📳 Assigned Haptic Pattern")
        st.json(pulse)
        
        # Visual indicator for vibration
        st.info(f"Triggering Pattern **{pulse.get('pattern_id')}**: {pulse.get('pulse_count')} pulse(s) of {pulse.get('pulse_duration_ms')}ms duration.")
    else:
        st.warning("No recent event found. Run your pulse bridge script first!")

st.markdown("---")
st.caption("Developed by Team Karna | VIT Bhopal University")
