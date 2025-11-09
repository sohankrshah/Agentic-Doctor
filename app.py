import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv
from src.main import run_diagnostic_pipeline, run_single_task
from src.chat_system.chat_interface import handle_ai_chat, export_chat_to_text
import uuid
from datetime import datetime
import json

load_dotenv()

st.set_page_config(
    page_title="AI Medical Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# Initialize session state
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""
if 'patient_age' not in st.session_state:
    st.session_state.patient_age = None
if 'case_id' not in st.session_state:
    st.session_state.case_id = str(uuid.uuid4())[:8]
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'uploaded_lab' not in st.session_state:
    st.session_state.uploaded_lab = None
if 'user_location' not in st.session_state:
    st.session_state.user_location = ""
if 'emergency_contacts' not in st.session_state:
    st.session_state.emergency_contacts = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

Path("uploads").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# Sidebar
with st.sidebar:  
    st.subheader("👤 Patient Information")
    
    patient_name = st.text_input("Patient Name", value=st.session_state.patient_name, placeholder="Enter full name")
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.patient_age if st.session_state.patient_age else 0)
    
    if st.button("🚀 Start Session", type="primary", use_container_width=True):
        if patient_name and patient_age > 0:
            st.session_state.patient_name = patient_name
            st.session_state.patient_age = patient_age
            st.session_state.case_id = str(uuid.uuid4())[:8]
            st.success(f"✅ Session started for {patient_name}")
            st.rerun()
        else:
            st.error("⚠️ Please enter name and age")
    
    if st.session_state.patient_name:
        st.divider()
        st.success("📋 **Active Session**")
        st.info(f"**👤 Name:** {st.session_state.patient_name}")
        st.info(f"**🎂 Age:** {st.session_state.patient_age} years")
        st.info(f"**🔖 Case ID:** {st.session_state.case_id}")
        
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.patient_name = ""
            st.session_state.patient_age = None
            st.session_state.case_id = str(uuid.uuid4())[:8]
            st.session_state.chat_history = []
            st.session_state.uploaded_image = None
            st.session_state.uploaded_lab = None
            st.session_state.user_input = ""
            st.rerun()

# Main header
st.title("🏥 AI Medical Assistant")
st.caption("Comprehensive Medical Triage with Evidence-Based Recommendations")

if not st.session_state.patient_name or not st.session_state.patient_age:
    st.warning("👈 Please enter patient information in the sidebar to begin")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "💬 Dr.AI Assistant Chat",
    "🚨 Emergency Support",
    "👥 Doctor Collaboration",
])

## TAB 1: Chat Interface
with tab1:
    st.header("💬 Dr.AI Assistant")

    # Set patient info if available
    if st.session_state.patient_name and st.session_state.patient_age:
        from src.chat_system.chat_interface import set_patient_info
        set_patient_info(
            st.session_state.case_id,
            st.session_state.patient_name,
            st.session_state.patient_age
        )

    # Display chat history
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            content_str = str(message['content']) if not isinstance(message['content'], str) else message['content']
            if message['role'] == 'user':
                with st.chat_message("user", avatar="👤"):
                    st.write(content_str)
            else:
                with st.chat_message("assistant", avatar="👨‍⚕️"):
                    st.markdown(content_str)
    else:
        st.info(f"👋 Welcome, {st.session_state.patient_name}! I'm Dr. Chen, your AI medical assistant. How are you feeling today?")

    st.divider()

    # Chat input form
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_area(
            "💭 Your message to Dr. Chen:",
            height=120,
            placeholder="Describe your symptoms or concerns...",
            help="Be as detailed as possible about your symptoms",
            key="chat_input_field"
        )

        # Unified button row inside the form
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            send_button = st.form_submit_button("📤 Send Message", type="primary", use_container_width=True)

        with col2:
            clear_button = st.form_submit_button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_btn")

        with col3:
            export_button = st.form_submit_button("💾 Export", use_container_width=True, key="export_chat_btn")

    # Handle Send Message
    if send_button and user_input.strip():
        with st.spinner("🤔 Dr. Chen is thinking..."):
            try:
                response = handle_ai_chat(user_input, st.session_state.case_id)
                response_text = str(response) if not isinstance(response, str) else response

                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now().isoformat()
                })
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response_text,
                    'timestamp': datetime.now().isoformat()
                })

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔧 Troubleshooting"):
                    st.markdown("""
                    **Common Issues:**

                    1. **Quota Exceeded (429 Error)**
                       - Check OpenAI billing: https://platform.openai.com/account/billing
                       - Add credits to your account

                    2. **Invalid API Key (401 Error)**
                       - Check your .env file
                       - Verify: `OPENAI_API_KEY=sk-proj-...`
                       - Get new key: https://platform.openai.com/api-keys

                    3. **Network Issues**
                       - Check internet connection
                       - Try again in a few moments
                    """)

    # Handle Clear Chat
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.user_input = ""
        st.rerun()

    # Handle Export
    if export_button and st.session_state.get("chat_history"):
        result = export_chat_to_text(st.session_state.case_id)
        st.success(result)
# TAB 2: Emergency Support
with tab2:
    st.header("🚨 Emergency Support Center")
    st.error("⚠️ In case of life-threatening emergency, call 112 (India) immediately")

    col1, col2 = st.columns(2)

    # 📞 Emergency Hotlines
    with col1:
        st.subheader("📞 Emergency Hotlines")
        emergency_numbers = {
            "Emergency Services": "112",
            "Ambulance": "102 / 108",
            "Police": "100",
            "Fire Brigade": "101",
            "Women Helpline": "1091",
            "Child Helpline": "1098",
            "Medical Emergency": "104",
            "Poison Control": "1800-425-2255"
        }

        for service, number in emergency_numbers.items():
            st.info(f"**{service}:** {number}")

        st.divider()

        # 🏥 Hospital Finder
        st.subheader("🏥 Find Nearest Hospital")
        user_city = st.text_input("City/Location", placeholder="e.g., Bhubaneswar")
        hospital_type = st.selectbox("Hospital Type", ["All", "Government", "Private", "Emergency", "Trauma", "Specialty"])
        search_radius = st.slider("Search Radius (km)", 1, 50, 10)

        if st.button("🔍 Search Hospitals", use_container_width=True):
            st.success(f"Searching for {hospital_type} hospitals within {search_radius}km of {user_city}")

    # 🚑 Ambulance Request
    with col2:
        st.subheader("🚑 Request Ambulance")
        location = st.text_input("Current Address")
        phone = st.text_input("Contact Number")
        emergency_type = st.selectbox("Emergency Type", [
            "Cardiac", "Accident", "Stroke", "Breathing Issue", "Bleeding", "Unconscious", "Poisoning", "Other"
        ])

        if st.button("🚑 Request Ambulance", type="primary", use_container_width=True):
            if location and phone:
                st.success("Ambulance request sent!")
                st.info(f"Location: {location}\nContact: {phone}\nType: {emergency_type}")
            else:
                st.error("Please enter both location and contact number")

        st.divider()

        # 🛏️ Bed Availability
        st.subheader("🛏️ Check Bed Availability")
        hospital = st.selectbox("Hospital", ["AIIMS", "Apollo", "Capital", "Sum", "Kalinga"])
        bed_type = st.selectbox("Bed Type", ["General", "ICU", "Emergency", "Maternity"])

        if st.button("🛏️ Check Beds", use_container_width=True):
            st.success(f"{bed_type} beds available at {hospital}")

# TAB 3: Doctor Collaboration
with tab3:
    st.header("👥 Doctor Collaboration")
    
    st.info("🚧 **Coming Soon:** Real-time collaboration with medical professionals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🩺 Features Coming Soon")
        st.write("- Video consultation with doctors")
        st.write("- Share medical records securely")
        st.write("- Second opinion requests")
        st.write("- Specialist referrals")
        st.write("- Follow-up scheduling")
    
    with col2:
        st.subheader("👨‍⚕️ Request Consultation")
        st.write("Connect with verified healthcare professionals for expert medical advice")

# Footer
st.divider()

st.markdown("""
**⚠️ Disclaimer:** This AI tool is for informational use only and does **not** replace professional medical advice.  
If symptoms worsen or in emergencies, call **112** (India) or consult a licensed doctor immediately.

**Emergency Numbers (India):** 112 | 102/108 (Ambulance)
""")

st.caption(f"AgenticDoctor v2.0 — Case ID: {st.session_state.case_id} | Powered by CrewAI & OpenAI | © 2025")