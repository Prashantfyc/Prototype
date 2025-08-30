# app.py
"""
Medico: Your Digital Mental Health Companion (v6.0 - Gamification)
- Added 'Achievements' page to track usage and award badges.
"""
import os
import re
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from streamlit_option_menu import option_menu
import numpy as np

# --- Configuration & Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Gemini Integration & AI Functions ---
model = genai.GenerativeModel('gemini-1.5-flash')
SYSTEM_PROMPT = """You are Medico, a supportive, empathetic, and non-judgmental digital companion... (rest of prompt is unchanged)"""

KNOWLEDGE_DOCUMENTS = [
    {"title": "On-Campus Doctor", "content": "..."},
    {"title": "Campus Counseling Center", "content": "..."},
    {"title": "Emergency Services", "content": "..."},
    {"title": "Exam Stress Tips", "content": "..."},
    {"title": "Dealing with Loneliness", "content": "..."},
    {"title": "Self-Care Strategies", "content": "..."},
] # Redacted for brevity

def find_best_match(query, documents):
    # (Function is unchanged)
    try:
        query_embedding = genai.embed_content(model='models/embedding-001', content=query, task_type="RETRIEVAL_QUERY")["embedding"]
        doc_embeddings = genai.embed_content(model='models/embedding-001', content=[doc['content'] for doc in documents], task_type="RETRIEVAL_DOCUMENT")["embedding"]
        products = np.dot(np.array(doc_embeddings), np.array(query_embedding))
        index = np.argmax(products)
        CONFIDENCE_THRESHOLD = 0.65
        return documents[index] if products[index] > CONFIDENCE_THRESHOLD else None
    except Exception as e:
        st.error(f"Error finding relevant information: {e}")
        return None

def analyze_sentiment(text):
    # (Function is unchanged)
    try:
        prompt = f"""Analyze the sentiment of the following user message. Classify it as one of the following: "Positive", "Negative", "Anxious", "Neutral", or "Crisis-level Distress". Message: "{text}" Sentiment:"""
        response = model.generate_content(prompt)
        sentiment = response.text.strip().replace('"', '')
        return sentiment
    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
        return "Neutral"

def get_gemini_response(user_text, chat_history):
    # (Function is unchanged)
    sentiment = analyze_sentiment(user_text)
    relevant_doc = find_best_match(user_text, KNOWLEDGE_DOCUMENTS)
    context = ""
    if relevant_doc:
        context += f"Relevant Info from '{relevant_doc['title']}': {relevant_doc['content']}\n"
    context += f"The user's current sentiment appears to be: {sentiment}."
    history_formatted = ""
    for message in chat_history[-4:]:
        role = "User" if message["role"] == "user" else "Medico"
        history_formatted += f'{role}: {message["content"]}\n'
    prompt = (f"{SYSTEM_PROMPT}\n\nCONTEXT FOR YOUR RESPONSE:\n{context}\n\nCONVERSATION HISTORY (summary):\n{history_formatted}\n\nNEW MESSAGE:\nUser: {user_text}\n\nMedico:")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        return "I'm having a little trouble connecting right now. Please try again in a moment."

def detect_crisis(text: str) -> bool:
    # (Function is unchanged)
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ["kill myself", "end my life", "want to die", "suicide", "hurt myself", "self harm"])

def log_event(filename, data):
    # (Function is unchanged)
    try:
        df = pd.DataFrame([data])
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
    except Exception as e:
        st.error(f"Failed to log event: {e}")

# --- New: Gamification Functions ---
def load_user_stats(stats_file):
    """Loads user stats from a JSON file, creating it if it doesn't exist."""
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    else:
        return {"mood_logs": 0, "journal_entries": 0}

def save_user_stats(stats_file, stats):
    """Saves user stats to a JSON file."""
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats, f)

def increment_stat(stats_file, stat_name):
    """Loads, increments, and saves a specific user stat."""
    stats = load_user_stats(stats_file)
    stats[stat_name] = stats.get(stat_name, 0) + 1
    save_user_stats(stats_file, stats)

# --- UI Configuration & Styling ---
st.set_page_config(page_title="Medico", layout="wide", page_icon="🩺")

st.markdown("""
<style>
    /* (CSS is unchanged, redacted for brevity) */
    .stApp { background-color: #1E1E2E; color: #CDD6F4; }
    h1, h2, h3, h4, h5, h6 { color: #CDD6F4; }
    [data-testid="stSidebar"] { background-color: #181825; border-right: 1px solid #313244; }
    .st-emotion-cache-1c7y2kd { background-color: #89B4FA; border-radius: 20px 20px 5px 20px; color: #1E1E2E; align-self: flex-end; max-width: 70%; }
    .st-emotion-cache-4k6c3l { background-color: #313244; border-radius: 20px 20px 20px 5px; color: #CDD6F4; align-self: flex-start; max-width: 70%; }
    [data-testid="stChatInput"] { background-color: #181825; border-top: 1px solid #313244; }
    [data-testid="stChatInput"] input { color: #CDD6F4; }
    .stButton>button { background-color: #89B4FA; color: #1E1E2E; border: none; border-radius: 8px; }
    .stButton>button:hover { background-color: #74C7EC; color: #1E1E2E; }
    /* New styles for achievement badges */
    .badge-unlocked { border: 2px solid #89B4FA; background-color: #313244; padding: 15px; border-radius: 10px; text-align: center; }
    .badge-locked { border: 2px solid #45475A; background-color: #181825; padding: 15px; border-radius: 10px; text-align: center; opacity: 0.6; }
    .badge-emoji { font-size: 40px; }
    .badge-title { font-size: 18px; font-weight: bold; color: #CDD6F4; }
    .badge-desc { font-size: 14px; color: #BAC2DE; }
</style>
""", unsafe_allow_html=True)


# --- Main App Logic ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

if not st.session_state.logged_in:
    st.title("Welcome to Medico 🩺")
    st.write("Your personal mental health companion.")
    with st.form("login_form"):
        username = st.text_input("Please enter your name to begin")
        submitted = st.form_submit_button("Start Session")
        if submitted and username:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
    st.stop()

# --- Main Application (after login) ---
USER_DATA_DIR = f"user_data/{st.session_state.username}"
STATS_FILE = f"{USER_DATA_DIR}/stats.json"
MOOD_LOG_FILE = f"{USER_DATA_DIR}/mood_log.csv"
JOURNAL_FILE = f"{USER_DATA_DIR}/journal_entries.csv"

with st.sidebar:
    st.title(f"Hi, {st.session_state.username}!")
    st.write("Your friendly mental health companion.")
    page = option_menu(
        None, ["Chat", "Journal", "Mood Tracker", "Achievements", "Guided Exercises", "Resources"],
        icons=['chat-dots-fill', 'pencil-square', 'graph-up-arrow', 'trophy-fill', 'activity', 'info-circle-fill'],
        menu_icon="cast", default_index=0,
        styles={"container": {"background-color": "#181825"}, "nav-link-selected": {"background-color": "#313244"}}
    )
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.rerun()

# --- Page Content ---
if page == "Chat":
    # (Page content is unchanged, redacted for brevity)
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
            crisis_response = ( "..." )
            # (crisis response logic)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Medico is thinking..."):
                    response = get_gemini_response(prompt, st.session_state.messages)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

elif page == "Journal":
    st.header("📝 My Private Journal")
    st.write("A space for your thoughts.")
    journal_entry = st.text_area("Write a new journal entry...", height=250, label_visibility="collapsed")
    if st.button("Save Entry"):
        if journal_entry:
            log_event(JOURNAL_FILE, {"timestamp": datetime.utcnow().isoformat(), "entry": journal_entry})
            increment_stat(STATS_FILE, "journal_entries") # Increment stat
            st.success("Your journal entry has been saved!")
            st.rerun()
        else:
            st.warning("Please write something before saving.")
    # (Past entries logic is unchanged)
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
        increment_stat(STATS_FILE, "mood_logs") # Increment stat
        st.success(f"Mood logged as {mood_score}/10. Keep it up!")
    # (Mood graph logic is unchanged)
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

    # Define all possible badges
    badges = {
        "First Step": {"emoji": "👣", "desc": "Logged in for the first time.", "unlocked": True},
        "Journalist I": {"emoji": "✍️", "desc": "Write your first journal entry.", "unlocked": user_stats["journal_entries"] >= 1},
        "Reflective Writer": {"emoji": "📖", "desc": "Write 5 journal entries.", "unlocked": user_stats["journal_entries"] >= 5},
        "Mindful Observer I": {"emoji": "🧘", "desc": "Log your mood for the first time.", "unlocked": user_stats["mood_logs"] >= 1},
        "Consistent Check-in": {"emoji": "🗓️", "desc": "Log your mood 5 times.", "unlocked": user_stats["mood_logs"] >= 5},
        "Wellness Champion": {"emoji": "🥇", "desc": "Log your mood 10 times.", "unlocked": user_stats["mood_logs"] >= 10},
    }

    # Display badges in columns
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
            st.write("") # Add space
        col_index += 1

elif page == "Guided Exercises":
    # (Page content is unchanged, redacted for brevity)
    st.header("🧘 Guided Exercises")
    st.write("Take a moment for yourself with these short, guided exercises.")
    st.subheader("5-Minute Calming Breathing Exercise")
    st.video("https://www.youtube.com/watch?v=inpok4MKVLM")
    st.subheader("10-Minute Guided Mindfulness Meditation")
    st.video("https://www.youtube.com/watch?v=O-6f5wQXSu8")

elif page == "Resources":
    # (Page content is unchanged, redacted for brevity)
    st.header("📚 Wellness Resources")
    st.write("Here are some university-approved resources to support you.")
    for doc in KNOWLEDGE_DOCUMENTS:
        st.subheader(doc["title"])
        if "Emergency" in doc["title"]: st.error(doc["content"])
        elif "Counseling" in doc["title"] or "Doctor" in doc["title"]: st.info(doc["content"])
        else: st.success(doc["content"])

st.caption("Disclaimer: Medico is an AI prototype for hackathon demonstration...")