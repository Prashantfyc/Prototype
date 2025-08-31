# app.py
"""
Medico: Your Digital Mental Health Companion (v11.0 - Gemini API Final)
- Configured to run exclusively on the Google Gemini API.
"""
import os
import re
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
import numpy as np
import google.generativeai as genai

# --- Configuration & Setup ---
load_dotenv()

# --- Configure Gemini API ---
try:
    # Locally, this will use the GEMINI_API_KEY from your .env file.
    # When deployed on Streamlit Cloud, it uses the key from Secrets.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except (AttributeError, AssertionError, Exception) as e:
    st.error(f"Error configuring Gemini API. Is your API key set correctly in your .env file or Streamlit Secrets? Error: {e}")
    st.stop()


# --- Load College Database ---
try:
    colleges_df = pd.read_csv("colleges_db.csv")
    COLLEGE_LIST = colleges_df["college_name"].tolist()
except FileNotFoundError:
    st.error("Fatal Error: `colleges_db.csv` not found. Please create it.")
    st.stop()

# --- System Prompt & Static Knowledge Base ---
SYSTEM_PROMPT = """You are Medico, a supportive, empathetic, and proactive wellness coach. Your purpose is to provide a safe space and help students build healthy habits.

Your core roles are:
1.  **Listen and Support:** Act as a non-judgmental companion. Use the provided context (college info, conversation history) to be helpful.
2.  **Identify Opportunities:** If a user mentions a recurring problem (e.g., stress, procrastination, feeling overwhelmed) over several messages, or expresses a desire to change, identify this as an opportunity to help.
3.  **Propose Action Plans:** Proactively offer to create a simple, manageable action plan.
    - **Always ask for permission first.** (e.g., "It sounds like managing time is a real challenge right now. Would you be open to creating a small, simple plan to tackle it?")
    - **If they agree, suggest a SMART goal.** The goal should be Specific, Measurable, Achievable, Relevant, and Time-bound. (e.g., "For the next two days, how about we try using the Pomodoro Technique for just one 25-minute study session? I can remind you if you'd like.")
    - **Keep it simple.** Focus on one small habit at a time.

- **Boundaries:** You MUST use the specific college details provided. You are NOT a therapist. NEVER give diagnoses or medical advice.
"""

KNOWLEDGE_DOCUMENTS = [
    {"title": "Emergency Services", "content": "If you or someone you know is in immediate danger or a crisis, please don't wait. Call the National Emergency Helpline at 112 or your local emergency number."},
    {"title": "Exam Stress Tips", "content": "Feeling overwhelmed by exams is normal. Techniques like the Pomodoro method, staying hydrated, and getting enough sleep can help manage pressure."},
    {"title": "Dealing with Loneliness", "content": "Feeling lonely is common. Consider joining a student club, attending campus events, or volunteering to connect with others."},
]

# --- Gemini Core Function ---
# --- Gemini Core Function ---
# --- Gemini Core Function ---
def get_gemini_response(user_text, chat_history, college_info):
    """Generates a response from the Gemini API."""
    # New: Explicitly add the user's college to the context
    context = f"The user is a student at {college_info['college_name']}.\n\n"

    # Check for health-related keywords before adding specific details
    health_keywords = ["doctor", "counselor", "appointment", "checkup", "sick", "health", "therapist", "medical"]
    if any(keyword in user_text.lower() for keyword in health_keywords):
        context += "RELEVANT COLLEGE INFO:\n"
        context += f"Counselor: {college_info['counselor_name']}, Location: {college_info['counselor_location']}, Phone: {college_info['counselor_phone']}\n"
        context += f"Doctor: {college_info['doctor_name']}, Location: {college_info['doctor_location']}, Phone: {college_info['doctor_phone']}\n\n"

    # Check for general keywords
    for doc in KNOWLEDGE_DOCUMENTS:
        if any(word.lower() in user_text.lower() for word in doc["title"].split()):
            context += f"GENERAL INFO on '{doc['title']}': {doc['content']}\n"

    history_formatted = ""
    for message in chat_history[-4:]:
        role = "User" if message["role"] == "user" else "Model"
        history_formatted += f"**{role}:** {message['content']}\n"
    
    prompt = (f"{SYSTEM_PROMPT}\n\n"
              f"---CONTEXT---\n{context}\n\n"
              f"---CONVERSATION HISTORY---\n{history_formatted}\n\n"
              f"---NEW MESSAGE---\n**User:** {user_text}\n\n**Model:**")
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the Gemini API: {e}")
        return "I'm having a little trouble connecting right now. Please try again in a moment."

# --- Other Core Functions ---
CRISIS_KEYWORDS = ["kill myself", "end my life", "want to die", "suicide", "hurt myself", "self harm"]
def detect_crisis(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def log_event(filename, data):
    try:
        df = pd.DataFrame([data])
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
    except Exception as e:
        st.error(f"Failed to log event: {e}")

# --- Gamification Functions ---
def load_user_stats(stats_file):
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    else:
        return {"mood_logs": 0, "journal_entries": 0}

def save_user_stats(stats_file, stats):
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats, f)

def increment_stat(stats_file, stat_name):
    stats = load_user_stats(stats_file)
    stats[stat_name] = stats.get(stat_name, 0) + 1
    save_user_stats(stats_file, stats)

# --- UI & Main App Logic ---
st.set_page_config(page_title="Medico", layout="wide", page_icon="🩺")
st.markdown("""<style> ... </style>""", unsafe_allow_html=True) # Redacted for brevity

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'college_info' not in st.session_state:
    st.session_state.college_info = None

if not st.session_state.logged_in:
    st.title("Welcome to Medico 🩺")
    st.write("Your personal mental health companion.")
    with st.form("login_form"):
        username = st.text_input("Please enter your name")
        college = st.selectbox("Select your college", COLLEGE_LIST)
        submitted = st.form_submit_button("Start Session")
        if submitted and username and college:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.college_info = colleges_df[colleges_df["college_name"] == college].iloc[0].to_dict()
            st.rerun()
    st.stop()

USER_DATA_DIR = f"user_data/{st.session_state.college_info['college_name']}/{st.session_state.username}"
STATS_FILE = f"{USER_DATA_DIR}/stats.json"
MOOD_LOG_FILE = f"{USER_DATA_DIR}/mood_log.csv"
JOURNAL_FILE = f"{USER_DATA_DIR}/journal_entries.csv"

with st.sidebar:
    st.title(f"Hi, {st.session_state.username}!")
    st.write(f"_{st.session_state.college_info['college_name']}_")
    
    st.sidebar.warning("⚠️ **Prototype Version**\nThis is a hackathon prototype. It is not a substitute for professional medical advice.")

    page = option_menu(
        None, ["Chat", "Journal", "Mood Tracker", "Achievements", "Guided Exercises", "Resources"],
        icons=['chat-dots-fill', 'pencil-square', 'graph-up-arrow', 'trophy-fill', 'activity', 'info-circle-fill'],
        menu_icon="cast", default_index=0
    )
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.session_state.college_info = None
        st.rerun()

if page == "Chat":
    st.header("How are you feeling today?")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"Hi {st.session_state.username}! I'm Medico. What's on your mind?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if detect_crisis(prompt):
            crisis_response = (
                "🚨 It sounds like you are going through a very difficult time. **Your safety is the most important thing.** "
                "Please reach out to a professional right now. You are not alone.\n\n"
                "- **National Emergency Helpline:** 112\n\n"
                "Help is available, and there are people who want to support you."
            )
            with st.chat_message("assistant"):
                st.warning(crisis_response)
            st.session_state.messages.append({"role": "assistant", "content": crisis_response})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Medico is thinking..."):
                    response = get_gemini_response(prompt, st.session_state.messages, st.session_state.college_info)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

elif page == "Journal":
    st.header("📝 My Private Journal")
    st.write("A space for your thoughts.")
    journal_entry = st.text_area("Write a new journal entry...", height=250, label_visibility="collapsed")
    if st.button("Save Entry"):
        if journal_entry:
            log_event(JOURNAL_FILE, {"timestamp": datetime.utcnow().isoformat(), "entry": journal_entry})
            increment_stat(STATS_FILE, "journal_entries")
            st.success("Your journal entry has been saved!")
            st.rerun()
        else:
            st.warning("Please write something before saving.")
    if os.path.exists(JOURNAL_FILE):
        st.write("---")
        st.subheader("Past Entries")
        journal_df = pd.read_csv(JOURNAL_FILE)
        for index, row in reversed(list(journal_df.iterrows())):
            with st.expander(f"Entry from {pd.to_datetime(row['timestamp']).strftime('%B %d, %Y %H:%M')} UTC"):
                st.write(row['entry'])

elif page == "Mood Tracker":
    st.header("📊 Mood Tracker")
    st.write("How are you feeling today?")
    mood_score = st.slider("Rate your mood (1 = Very Down, 10 = Excellent)", 1, 10, 5)
    if st.button("Log My Mood"):
        log_event(MOOD_LOG_FILE, {"timestamp": datetime.utcnow().isoformat(), "mood_score": mood_score})
        increment_stat(STATS_FILE, "mood_logs")
        st.success(f"Mood logged as {mood_score}/10. Keep it up!")
    if os.path.exists(MOOD_LOG_FILE):
        st.write("---")
        st.subheader("Your Mood Over Time")
        mood_df = pd.read_csv(MOOD_LOG_FILE)
        mood_df['timestamp'] = pd.to_datetime(mood_df['timestamp'])
        if not mood_df.empty:
            fig = px.line(mood_df, x='timestamp', y='mood_score', markers=True, template="plotly_dark")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

elif page == "Achievements":
    st.header("🏆 Your Achievements")
    st.write("Celebrate your progress! Every step you take for your well-being is a victory.")
    user_stats = load_user_stats(STATS_FILE)
    st.write("---")
    badges = {
        "First Step": {"emoji": "👣", "desc": "Logged in for the first time.", "unlocked": True},
        "Journalist I": {"emoji": "✍️", "desc": "Write your first journal entry.", "unlocked": user_stats.get("journal_entries", 0) >= 1},
        "Reflective Writer": {"emoji": "📖", "desc": "Write 5 journal entries.", "unlocked": user_stats.get("journal_entries", 0) >= 5},
        "Mindful Observer I": {"emoji": "🧘", "desc": "Log your mood for the first time.", "unlocked": user_stats.get("mood_logs", 0) >= 1},
        "Consistent Check-in": {"emoji": "🗓️", "desc": "Log your mood 5 times.", "unlocked": user_stats.get("mood_logs", 0) >= 5},
        "Wellness Champion": {"emoji": "🥇", "desc": "Log your mood 10 times.", "unlocked": user_stats.get("mood_logs", 0) >= 10},
    }
    cols = st.columns(3)
    col_index = 0
    for title, data in badges.items():
        with cols[col_index % 3]:
            badge_class = "badge-unlocked" if data["unlocked"] else "badge-locked"
            st.markdown(f"""
            <div class="{badge_class}">
                <div class="badge-emoji">{data["emoji"]}</div>
                <div class="badge-title">{title}</div>
                <div class="badge-desc">{data["desc"]}</div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
        col_index += 1

elif page == "Guided Exercises":
    st.header("🧘 Guided Exercises")
    st.write("Take a moment for yourself with these short, guided exercises.")
    st.subheader("5-Minute Calming Breathing Exercise")
    st.video("https://www.youtube.com/watch?v=inpok4MKVLM")
    st.subheader("10-Minute Guided Mindfulness Meditation")
    st.video("https://www.youtube.com/watch?v=O-6f5wQXSu8")

elif page == "Resources":
    st.header("📚 Wellness Resources")
    st.write("Here are some university-approved resources to support you.")
    for doc in KNOWLEDGE_DOCUMENTS:
        st.subheader(doc["title"])
        if "Emergency" in doc["title"]: st.error(doc["content"])
        elif "Counseling" in doc["title"] or "Doctor" in doc["title"]: st.info(doc["content"])
        else: st.success(doc["content"])