# app.py
"""
Medico: Your Digital Mental Health Companion (v3.0 - Advanced RAG)
- Upgraded to use embeddings for smarter, meaning-based knowledge retrieval.
"""
import os
import re
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

# --- File paths for data storage ---
LOG_FILE = "flagged_chats.csv"
MOOD_LOG_FILE = "mood_log.csv"
JOURNAL_FILE = "journal_entries.csv"

# --- Gemini Integration ---
model = genai.GenerativeModel('gemini-1.5-flash')
SYSTEM_PROMPT = """You are Medico, a supportive, empathetic, and non-judgmental digital companion from the university's wellness department. Your purpose is to provide a safe and encouraging space for students.
Your role is to:
- Listen actively and respond with compassion and understanding.
- Provide evidence-based self-care tips, stress-management techniques (like mindfulness, breathing exercises), and positive coping strategies.
- Gently encourage users to explore features like the journal and mood tracker for self-reflection.
- If a user mentions a specific problem (e.g., exam stress, loneliness), retrieve and present relevant information from the knowledge base in a helpful, conversational way.
- Maintain a warm, hopeful, and respectful tone.
- CRUCIAL: You are NOT a therapist. You must NEVER provide diagnoses, medical advice, or prescriptions. You are a supportive peer.
- If you detect a crisis (mentions of self-harm, suicide, etc.), immediately trigger the safety protocol by providing the predefined crisis message and disengaging from further conversation on that topic.
"""

# --- New: Advanced Knowledge Base (RAG) using Embeddings ---
KNOWLEDGE_DOCUMENTS = [
    {
        "title": "On-Campus Doctor",
        "content": "The on-campus doctor is available at the University Health Center. Details: Dr. Anya Sharma (General Physician), Location: Health & Wellness Building, Ground Floor, Room 102. Phone: +91-ZZZZZZZZZZ. Hours: Mon-Fri 10:00-16:00. An appointment is recommended."
    },
    {
        "title": "Campus Counseling Center",
        "content": "Our Campus Counseling Center is a free and confidential resource for students needing to talk to a professional about stress, anxiety, or feeling down. Location: Health & Wellness Building, Room 204. Phone: +91-XXXXXXXXXX. They offer individual sessions, group therapy, and workshops."
    },
    {
        "title": "Emergency Services",
        "content": "If you or someone you know is in immediate danger or a crisis, please don't wait. Call the National Emergency Helpline at 112 or the Campus Security emergency line at +91-YYYYYYYYY. Help is available 24/7."
    },
    {
        "title": "Exam Stress Tips",
        "content": "Feeling overwhelmed by exams is normal. Techniques like the Pomodoro method (study for 25 mins, break for 5), staying hydrated, getting 7-8 hours of sleep, and light exercise can help manage pressure and improve focus."
    },
    {
        "title": "Dealing with Loneliness",
        "content": "Feeling lonely or isolated is a common experience. Consider joining a student club that matches your interests, attending campus events, or volunteering. The Student Life office has a full list of clubs. Small steps can make a big difference."
    },
    {
        "title": "Self-Care Strategies",
        "content": "Self-care is vital for well-being. Try simple things like a 5-minute guided meditation, journaling your thoughts, going for a walk, or listening to calming music. These small actions can have a big impact on your mental state."
    }
]

# --- New: Embedding and Retrieval Functions ---
def find_best_match(query, documents):
    """Finds the most relevant document from the knowledge base using embeddings."""
    try:
        query_embedding = genai.embed_content(model='models/embedding-001',
                                              content=query,
                                              task_type="RETRIEVAL_QUERY")["embedding"]

        doc_embeddings = genai.embed_content(model='models/embedding-001',
                                             content=[doc['content'] for doc in documents],
                                             task_type="RETRIEVAL_DOCUMENT")["embedding"]

        products = np.dot(np.array(doc_embeddings), np.array(query_embedding))
        index = np.argmax(products)
        
        CONFIDENCE_THRESHOLD = 0.65
        return documents[index] if products[index] > CONFIDENCE_THRESHOLD else None

    except Exception as e:
        st.error(f"Error finding relevant information: {e}")
        return None

def get_gemini_response(user_text, chat_history):
    """Generates a response from Gemini, including context and conversation history."""
    relevant_doc = find_best_match(user_text, KNOWLEDGE_DOCUMENTS)
    
    context = ""
    if relevant_doc:
        context = f"Relevant Info from '{relevant_doc['title']}': {relevant_doc['content']}"

    history_formatted = ""
    for message in chat_history[-4:]:
        role = "User" if message["role"] == "user" else "Medico"
        history_formatted += f'{role}: {message["content"]}\n'

    prompt = (f"{SYSTEM_PROMPT}\n\n"
              f"{context}\n\n"
              f"CONVERSATION HISTORY (summary):\n{history_formatted}\n\n"
              f"NEW MESSAGE:\nUser: {user_text}\n\nMedico:")

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        return "I'm having a little trouble connecting right now. Please try again in a moment."

# --- Core Functions (Unchanged) ---
CRISIS_KEYWORDS = ["kill myself", "end my life", "want to die", "suicide", "hurt myself", "self harm"]

def detect_crisis(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def log_event(filename, data):
    try:
        df = pd.DataFrame([data])
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
    except Exception as e:
        st.error(f"Failed to log event: {e}")

# --- UI Configuration & Styling ---
st.set_page_config(page_title="Medico", layout="wide", page_icon="🩺")

st.markdown("""
<style>
    /* --- Base App Style --- */
    .stApp {
        background-color: #1E1E2E; /* Dark blue-gray background */
        color: #CDD6F4; /* Light lavender text */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #CDD6F4;
    }

    /* --- Sidebar Style --- */
    [data-testid="stSidebar"] {
        background-color: #181825; /* Even darker sidebar */
        border-right: 1px solid #313244;
    }
    
    /* --- Chat Bubbles Style --- */
    .st-emotion-cache-1c7y2kd { /* User chat bubble */
        background-color: #89B4FA; /* User bubble blue */
        border-radius: 20px 20px 5px 20px;
        color: #1E1E2E; /* Dark text on light bubble */
        align-self: flex-end; /* Align to the right */
        max-width: 70%;
    }
    .st-emotion-cache-4k6c3l { /* Bot chat bubble */
        background-color: #313244; /* Bot bubble gray */
        border-radius: 20px 20px 20px 5px;
        color: #CDD6F4; /* Light text on dark bubble */
        align-self: flex-start; /* Align to the left */
        max-width: 70%;
    }

    /* --- Chat Input Box Style --- */
    [data-testid="stChatInput"] {
        background-color: #181825;
        border-top: 1px solid #313244;
    }
    [data-testid="stChatInput"] input {
        color: #CDD6F4;
    }

    /* --- Button & Widget Style --- */
    .stButton>button {
        background-color: #89B4FA;
        color: #1E1E2E;
        border: none;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #74C7EC;
        color: #1E1E2E;
    }

</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.title("🩺 Medico")
    st.write("Your friendly mental health companion.")

    page = option_menu(
        None, ["Chat", "Journal", "Mood Tracker", "Resources"],
        icons=['chat-dots-fill', 'pencil-square', 'graph-up-arrow', 'info-circle-fill'],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#181825"},
            "icon": {"color": "#CDD6F4", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color": "#CDD6F4"},
            "nav-link-selected": {"background-color": "#313244", "color": "#89B4FA"},
        }
    )

# --- Main Content Area ---
if page == "Chat":
    st.header("How are you feeling today?")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi there! I'm Medico. I'm here to offer support and a listening ear. What's on your mind?"}]

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
                "- **Campus Counseling Emergency:** +91-XXXXXXXXXX\n"
                "- **National Emergency Helpline:** 112\n\n"
                "Help is available, and there are people who want to support you."
            )
            with st.chat_message("assistant"):
                st.warning(crisis_response)
            st.session_state.messages.append({"role": "assistant", "content": crisis_response})
            log_event(LOG_FILE, {"timestamp": datetime.utcnow().isoformat(), "message": prompt})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Medico is thinking..."):
                    response = get_gemini_response(prompt, st.session_state.messages)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

elif page == "Journal":
    st.header("📝 My Private Journal")
    st.write("A space for your thoughts. Writing can be a powerful tool for clarity.")
    
    journal_entry = st.text_area("Write a new journal entry...", height=250, key="journal_text", label_visibility="collapsed")
    if st.button("Save Entry"):
        if journal_entry:
            log_event(JOURNAL_FILE, {"timestamp": datetime.utcnow().isoformat(), "entry": journal_entry})
            st.success("Your journal entry has been saved securely.")
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
    st.write("Checking in with yourself is a great habit. How are you feeling today?")
    
    mood_score = st.slider("Rate your mood (1 = Very Down, 10 = Excellent)", 1, 10, 5, label_visibility="collapsed")
    if st.button("Log My Mood"):
        log_event(MOOD_LOG_FILE, {"timestamp": datetime.utcnow().isoformat(), "mood_score": mood_score})
        st.success(f"Mood logged as {mood_score}/10. Keep it up!")
    
    if os.path.exists(MOOD_LOG_FILE):
        st.write("---")
        st.subheader("Your Mood Over Time")
        mood_df = pd.read_csv(MOOD_LOG_FILE)
        mood_df['timestamp'] = pd.to_datetime(mood_df['timestamp'])
        if not mood_df.empty:
            fig = px.line(mood_df, x='timestamp', y='mood_score', markers=True, title="Mood Trend", labels={'timestamp': 'Date', 'mood_score': 'Mood Score (1-10)'}, template="plotly_dark")
            fig.update_layout(yaxis=dict(range=[0,11]), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No mood data yet. Log your mood to see your trend!")

elif page == "Resources":
    st.header("📚 Wellness Resources")
    st.write("Here are some university-approved resources to support you.")
    
    # Updated to loop through the new KNOWLEDGE_DOCUMENTS structure
    for doc in KNOWLEDGE_DOCUMENTS:
        st.subheader(doc["title"])
        if "Emergency" in doc["title"]:
            st.error(doc["content"])
        elif "Counseling" in doc["title"] or "Doctor" in doc["title"]:
            st.info(doc["content"])
        else:
            st.success(doc["content"])


st.caption("Disclaimer: Medico is an AI prototype for hackathon demonstration and is not a substitute for professional medical advice.")