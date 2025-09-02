# app.py
"""
Medico: Digital Mental Health Companion (v12.0)
- Added comprehensive user profiling with emotional context
- Implemented emotion detection and personalized AI responses
- Enhanced UI with beautiful custom styling and animations
- Added response length moderation based on user profile
- Created stunning hackathon-worthy interface
"""

import os
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
import numpy as np
import google.generativeai as genai
from textblob import TextBlob
import time

# Configuration & Setup
load_dotenv()

# Enhanced Custom CSS for Stunning UI
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }

    /* Fix the entire app background to a dark color */
    .stApp {
        background: #1e1e2f !important;
        color: white !important;
        background-attachment: fixed;
    }

    /* Fix the main container background color */
    .block-container {
        background-color: #1e1e2f !important;
        color: white !important;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Fix all page headers and section titles to white */
    h1, h2, h3, h4, h5, h6 {
         background-color: #1e1e2f !important;
        color: white !important;
    }
    
    /* Fix label colors to white for visibility */
    label, .stTextInput label, .stSelectbox label {
        background-color: #1e1e2f !important;
        color: white !important;
        font-weight: 600;
    }

 

    /* Make the form background darker for better label contrast */
    .login-container {
        background: linear-gradient(135deg, #363B49 0%, #764BA2 100%);
        color: #fff;
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 0 25px 50px rgba(0,0,0,0.15), 0 0 0 1px rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        max-width: 800px;
        margin: 2rem auto;
        animation: slideInUp 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
                /* Sidebar with glassmorphism and gradient shadow */
div[data-testid="stSidebar"] {
    background: linear-gradient(135deg, rgba(45,46,66,0.94) 60%, #764ba2 120%);
    box-shadow: 0 8px 32px 0 rgba(56,44,119,0.24);
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(18px);
    min-width: 230px;
    color: #fff;
}

/* Profile Section */
.profile-section {
    background: linear-gradient(90deg, #2d2e42 60%, #473e7f 120%);
    border-radius: 18px;
    padding: 1.8rem 1rem 1rem 1rem;
    margin-bottom: 18px;
    box-shadow: 0 4px 24px rgba(118,75,162,0.14);
}

.profile-section h2 {
    font-weight: 700;
    font-size: 1.6rem;
    margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #b993d6 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.profile-section em {
    color: #c7bacb;
    font-size: 1.01rem;
    margin-bottom: 0.8rem;
    font-style: italic;
}

/* Badges with gradient and glow */
.profile-section span {
    background: linear-gradient(90deg, #764ba2 10%, #b993d6 80%);
    color: #fff;
    font-weight: 600;
    font-size: 0.87rem;
    border-radius: 12px;
    padding: 0.37rem 1.2rem;
    margin-right: 0.4rem;
    margin-bottom: 0.35rem;
    box-shadow: 0 2px 12px rgba(113,77,255,0.18);
    display: inline-block;
}

.profile-section span:last-child {
    margin-right: 0;
}

/* Navigation items: icons and text */
.st-emotion-cache-17z4y3l, .st-emotion-cache-6kekos, .nav-link, .nav-link-selected {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #fff !important;
    padding: 0.9rem 1.1rem 0.9rem 1.1rem !important;
    margin-bottom: 7px !important;
    transition: box-shadow 0.2s;
}

/* Active nav (selected page) with glow effect */
.st-emotion-cache-17z4y3l[aria-selected="true"], .nav-link-selected {
    background: linear-gradient(90deg, #8462d7 45%, #a67be0 120%);
    color: #fff !important;
    box-shadow: 0 0 10px 3px rgba(118,75,162,0.33);
}

/* Navigation icons - larger, colored, animated on hover */
.st-emotion-cache-18ni7ap svg, .nav-link svg {
    font-size: 1.33em !important;
    transition: transform 0.15s;
    color: #b993d6 !important;
}

.nav-link:hover svg, .st-emotion-cache-17z4y3l:hover svg {
    transform: scale(1.12);
    color: #f5e7ff !important;
}

/* Divider line */
hr {
    border: none;
    border-top: 1px solid #473e7f;
    margin: 16px 0;
}

/* Footer/team credits etc. in sidebar */
.stSidebarFooter {
    color: #c7bacb !important;
    font-size: 0.98rem;
    margin-top: 16px;
    text-align: center;
    opacity: 0.8;
}

    /* Other styles remain same here... */
    /* You can keep the rest of your existing CSS */
    
    </style>
    """, unsafe_allow_html=True)



# Configure Gemini API
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except (AttributeError, AssertionError, Exception) as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# Load College Database
try:
    colleges_df = pd.read_csv("colleges_db.csv")
    COLLEGE_LIST = colleges_df["college_name"].tolist()
except FileNotFoundError:
    st.error("Fatal Error: colleges_db.csv not found.")
    st.stop()

# Enhanced Emotion Detection
def detect_emotion(text):
    """Advanced emotion detection using TextBlob and keyword analysis"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Comprehensive emotion keywords
    emotions = {
        'joy': ['happy', 'excited', 'great', 'awesome', 'amazing', 'love', 'wonderful', 'fantastic', 'thrilled', 'delighted'],
        'sadness': ['sad', 'depressed', 'down', 'upset', 'crying', 'hurt', 'lonely', 'empty', 'devastated', 'heartbroken'],
        'anger': ['angry', 'mad', 'furious', 'irritated', 'frustrated', 'rage', 'annoyed', 'livid', 'outraged'],
        'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'panic', 'terrified', 'frightened', 'apprehensive'],
        'stress': ['stressed', 'overwhelmed', 'pressure', 'tension', 'burden', 'exhausted', 'burnout', 'overloaded'],
        'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow', 'incredible'],
        'neutral': ['okay', 'fine', 'normal', 'average', 'regular', 'alright']
    }
    
    text_lower = text.lower()
    detected_emotions = []
    
    # Check for emotion keywords
    for emotion, keywords in emotions.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_emotions.append(emotion)
    
    # Determine primary emotion based on polarity and keywords
    if detected_emotions:
        primary_emotion = detected_emotions[0]
    elif polarity > 0.5:
        primary_emotion = 'joy'
    elif polarity < -0.5:
        primary_emotion = 'sadness'
    elif polarity < -0.1:
        primary_emotion = 'anger'
    elif subjectivity > 0.7:
        primary_emotion = 'stress'
    else:
        primary_emotion = 'neutral'
    
    return {
        'primary_emotion': primary_emotion,
        'emotions': detected_emotions,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'intensity': abs(polarity)
    }

# Personalized System Prompt Generator
def get_personalized_system_prompt(user_profile, emotion_data):
    """Generate highly personalized AI system prompt based on user profile and emotions"""
    
    base_prompt = """You are Medico, an advanced AI wellness companion providing personalized mental health support."""
    
    # Age-based communication styles
    age_styles = {
        '18-25': "Use encouraging, relatable language. Reference common college experiences. Be detailed in explanations.",
        '26-35': "Professional yet warm tone. Acknowledge career and relationship pressures. Provide practical advice.",
        '36-45': "Respectful, efficient communication. Focus on actionable solutions for busy lifestyles.",
        '46-60': "Thoughtful, experienced tone. Consider life transitions and family responsibilities.",
        '60+': "Patient, respectful approach. Value their wisdom while offering gentle support."
    }
    
    # Education-based response complexity
    education_styles = {
        'High School': "Simple language, motivational tone. Focus on study habits and future planning.",
        'Undergraduate': "Balanced complexity. Address academic stress, social issues, and career anxiety.",
        'Graduate': "Sophisticated language. Time-efficient responses. Evidence-based suggestions.",
        'PhD/Research': "Concise, research-backed advice. Respect their expertise. 2-3 sentences maximum.",
        'Professional': "Direct, practical solutions. Focus on work-life balance and stress management."
    }
    
    # Career concern specializations
    career_responses = {
        'Exams': "Focus on study techniques, test anxiety, and performance optimization.",
        'Interviews': "Address confidence building, preparation strategies, and rejection resilience.",
        'Career Path': "Discuss goal setting, decision-making frameworks, and future planning.",
        'Job Security': "Address financial anxiety, skill development, and career stability.",
        'Work-Life Balance': "Focus on boundaries, time management, and personal well-being."
    }
    
    # Emotional response modulation
    emotion_approaches = {
        'joy': "Match their positive energy. Help maintain realistic optimism. Celebrate achievements.",
        'sadness': "Extra empathy and gentleness. Validate feelings. Suggest small, manageable steps.",
        'anger': "Acknowledge without judgment. Help process feelings constructively. Suggest outlets.",
        'fear': "Provide reassurance. Break overwhelming situations into manageable parts.",
        'stress': "Focus on immediate relief techniques. Prioritization strategies. Breathing exercises.",
        'surprise': "Acknowledge the unexpected. Help process and adapt to new situations.",
        'neutral': "Maintain warm engagement. Gently assess if more support is needed."
    }
    
    # Build personalized context
    prompt_parts = [base_prompt]
    
    # Add age-appropriate communication
    if user_profile.get('age_group'):
        prompt_parts.append(age_styles.get(user_profile['age_group'], ""))
    
    # Add education-based complexity
    if user_profile.get('education_level'):
        prompt_parts.append(education_styles.get(user_profile['education_level'], ""))
    
    # Add career-specific focus
    if user_profile.get('career_concerns'):
        for concern in user_profile['career_concerns']:
            if concern in career_responses:
                prompt_parts.append(career_responses[concern])
    
    # Add emotional context
    if emotion_data:
        emotion_approach = emotion_approaches.get(emotion_data['primary_emotion'], "")
        prompt_parts.append(f"User's emotional state: {emotion_data['primary_emotion']} (intensity: {emotion_data['intensity']:.2f}). {emotion_approach}")
    
    # Add response length guidance
    if user_profile.get('education_level') in ['PhD/Research', 'Graduate']:
        prompt_parts.append("CRITICAL: Keep responses concise and actionable (maximum 2-3 sentences).")
    elif user_profile.get('education_level') == 'High School':
        prompt_parts.append("Provide detailed, educational responses with examples and encouragement.")
    
    # Add demographic considerations
    if user_profile.get('financial_support') in ['Struggling', 'Very Difficult']:
        prompt_parts.append("Be sensitive to financial constraints. Suggest free or low-cost resources.")
    
    if user_profile.get('discrimination') in ['Often', 'Sometimes']:
        prompt_parts.append("Be extra sensitive to discrimination issues. Validate their experiences.")
    
    if user_profile.get('bullying') in ['Currently', 'In the past']:
        prompt_parts.append("Be gentle and supportive. Focus on building confidence and self-worth.")
    
    return "\n\n".join(filter(None, prompt_parts))

# Enhanced Response Length Modulation
def determine_response_length(user_input, user_profile, emotion_data):
    """Determine appropriate response length based on multiple factors"""
    input_length = len(user_input.split())
    
    # Base length determination
    if input_length <= 3:
        base_length = "very_short"
    elif input_length <= 10:
        base_length = "short"
    elif input_length <= 25:
        base_length = "medium"
    else:
        base_length = "long"
    
    # Adjust for user profile
    if user_profile.get('education_level') in ['PhD/Research', 'Graduate']:
        if base_length in ['short', 'very_short']:
            return "Respond in 1-2 sentences maximum. Be direct and actionable."
        else:
            return "Keep response focused and concise (maximum 3-4 sentences)."
    
    elif user_profile.get('age_group') == '18-25':
        if base_length == 'very_short':
            return "Provide a warm, encouraging response with some detail (3-4 sentences)."
        else:
            return "Give a comprehensive, supportive response with examples."
    
    # Adjust for emotional state
    if emotion_data and emotion_data['primary_emotion'] in ['sadness', 'fear', 'stress']:
        return "Prioritize empathy and validation. Provide gentle, supportive guidance."
    elif emotion_data and emotion_data['primary_emotion'] == 'anger':
        return "Acknowledge their feelings calmly. Be concise but understanding."
    
    return "Provide a balanced, helpful response appropriate to their needs."

# Enhanced Gemini Response Function
def get_enhanced_gemini_response(user_text, chat_history, college_info, user_profile):
    """Generate highly personalized AI response"""
    
    # Detect emotion
    emotion_data = detect_emotion(user_text)
    
    # Generate personalized system prompt
    system_prompt = get_personalized_system_prompt(user_profile, emotion_data)
    
    # Determine response length
    length_instruction = determine_response_length(user_text, user_profile, emotion_data)
    
    # Build comprehensive context
    context_parts = []
    context_parts.append(f"Student at: {college_info['college_name']}")
    context_parts.append(f"Profile: {user_profile['age_group']}, {user_profile['education_level']}")
    context_parts.append(f"Location: {user_profile['location_type']}")
    context_parts.append(f"Status: {user_profile['relationship_status']}")
    
    if user_profile.get('career_concerns'):
        context_parts.append(f"Concerns: {', '.join(user_profile['career_concerns'])}")
    
    context_parts.append(f"Emotional state: {emotion_data['primary_emotion']} (intensity: {emotion_data['intensity']:.2f})")
    
    # Add college resources for health keywords
    health_keywords = ["doctor", "counselor", "appointment", "checkup", "sick", "health", "therapist", "medical"]
    if any(keyword in user_text.lower() for keyword in health_keywords):
        context_parts.append("COLLEGE RESOURCES:")
        context_parts.append(f"Counselor: {college_info.get('counselor_name', 'Contact your college')}")
        context_parts.append(f"Doctor: {college_info.get('doctor_name', 'Contact campus health center')}")
    
    context = "\n".join(context_parts)
    
    # Format recent chat history
    history_formatted = ""
    for message in chat_history[-3:]:
        role = "User" if message["role"] == "user" else "Medico"
        history_formatted += f"{role}: {message['content'][:100]}...\n"
    
    # Construct final prompt
    final_prompt = f"""{system_prompt}
    
RESPONSE GUIDELINES: {length_instruction}

CONTEXT: {context}

RECENT CONVERSATION:
{history_formatted}

USER MESSAGE: {user_text}

MEDICO RESPONSE:"""
    
    try:
        response = gemini_model.generate_content(final_prompt)
        return response.text, emotion_data
    except Exception as e:
        st.error(f"AI Error: {e}")
        return "I'm having trouble connecting right now. Please try again in a moment.", emotion_data

# Crisis Detection (Enhanced)
CRISIS_KEYWORDS = [
    "kill myself", "end my life", "want to die", "suicide", "hurt myself", 
    "self harm", "no point living", "can't go on", "end it all", "not worth living"
]

def detect_crisis(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

# Utility Functions
def log_event(filename, data):
    try:
        df = pd.DataFrame([data])
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
    except Exception as e:
        st.error(f"Logging error: {e}")

def load_user_stats(stats_file):
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    return {"mood_logs": 0, "journal_entries": 0, "chat_messages": 0, "login_count": 0}

def save_user_stats(stats_file, stats):
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats, f)

def increment_stat(stats_file, stat_name):
    stats = load_user_stats(stats_file)
    stats[stat_name] = stats.get(stat_name, 0) + 1
    save_user_stats(stats_file, stats)

# Main App Configuration
st.set_page_config(
    page_title="Medico", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

load_custom_css()

# Session State Management
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'college_info' not in st.session_state:
    st.session_state.college_info = None

# Enhanced Login Form with Comprehensive Profiling
if not st.session_state.logged_in:
    st.markdown("""
    <div class='login-container'>
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='font-size: 3rem; margin-bottom: 0.5rem; background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🧠 Medico </h1>
           <p style='font-size: 1.2rem; color: white; margin-bottom: 2rem;'>Your Personalized AI Mental Health Companion</p>
<div style='width: 100px; height: 4px; background: linear-gradient(45deg, #667eea, #764ba2); margin: 0 auto; border-radius: 2px;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("comprehensive_profile_form"):
        # Basic Information
        st.subheader("Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Full Name", placeholder="Enter your name")
            college = st.selectbox("College/Institution", COLLEGE_LIST)
            age_group = st.selectbox("Age Group", 
                ['18-25', '26-35', '36-45', '46-60', '60+'])
            location_type = st.selectbox("Location Type", 
                ['Urban', 'Rural', 'Suburban'])
        
        with col2:
            education_level = st.selectbox("Education Level", 
                ['High School', 'Undergraduate', 'Graduate', 'PhD/Research', 'Professional'])
            relationship_status = st.selectbox("Relationship Status", 
                ['Single', 'In Relationship', 'Married', 'Divorced', 'Widowed', 'Prefer not to say'])
            financial_support = st.selectbox("Financial Situation", 
                ['Very Stable', 'Stable', 'Managing', 'Struggling', 'Very Difficult'])
            family_situation = st.selectbox("Family Support", 
                ['Very Supportive', 'Supportive', 'Neutral', 'Some Issues', 'Major Issues'])
        
        # Career & Academic Concerns
        st.subheader("Career & Academic Concerns")
        career_concerns = st.multiselect("Current Challenges", 
            ['Exams', 'Interviews', 'Career Path', 'Job Security', 'Work-Life Balance', 
             'Academic Pressure', 'Research Stress', 'Thesis/Dissertation'])
        
        # Environmental & Social Factors
        st.subheader("Environmental & Social Factors")
        col3, col4 = st.columns(2)
        
        with col3:
            environment_stressors = st.multiselect("Environment Stressors", 
                ['Academic Pressure', 'Social Media', 'News/Politics', 'Climate Anxiety', 
                 'Social Isolation', 'Work Environment', 'Living Situation'])
            discrimination = st.selectbox("Experienced Discrimination", 
                ['Never', 'Rarely', 'Sometimes', 'Often', 'Prefer not to say'])
        
        with col4:
            bullying = st.selectbox("Experienced Bullying", 
                ['Never', 'In the past', 'Currently', 'Prefer not to say'])
            
        # Privacy Notice
        st.info("Your information is stored locally and never shared. This helps personalize your AI companion.")
        
        # Enhanced Submit Button
        submitted = st.form_submit_button("Begin My Wellness Journey", use_container_width=True)
        
        if submitted and username and college:
            # Save comprehensive profile
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user_profile = {
                'age_group': age_group,
                'location_type': location_type,
                'relationship_status': relationship_status,
                'education_level': education_level,
                'career_concerns': career_concerns,
                'financial_support': financial_support,
                'family_situation': family_situation,
                'environment_stressors': environment_stressors,
                'discrimination': discrimination,
                'bullying': bullying,
                'registration_date': datetime.now().isoformat()
            }
            st.session_state.college_info = colleges_df[colleges_df["college_name"] == college].iloc[0].to_dict()
            
            # Increment login count
            st.success("Profile created successfully! Redirecting to your personalized dashboard...")
            time.sleep(2)
            st.rerun()
    
    st.stop()

# File Paths
USER_DATA_DIR = f"user_data/{st.session_state.college_info['college_name']}/{st.session_state.username}"
STATS_FILE = f"{USER_DATA_DIR}/stats.json"
MOOD_LOG_FILE = f"{USER_DATA_DIR}/mood_log.csv"
JOURNAL_FILE = f"{USER_DATA_DIR}/journal_entries.csv"
PROFILE_FILE = f"{USER_DATA_DIR}/profile.json"

# Save user profile
os.makedirs(USER_DATA_DIR, exist_ok=True)
with open(PROFILE_FILE, 'w') as f:
    json.dump(st.session_state.user_profile, f)

# Enhanced Sidebar with Profile Display
with st.sidebar:
    # Profile Header
    st.markdown(f"""
    <div class='profile-section'>
        <h2 style='margin: 0; font-size: 1.5rem;'>Hi, {st.session_state.username}!</h2>
        <p style='margin: 0.5rem 0; opacity: 0.9;'><em>{st.session_state.college_info['college_name']}</em></p>
        <div style='display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;'>
            <span style='background: rgba(255,255,255,0.2); padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;'>{st.session_state.user_profile['age_group']}</span>
            <span style='background: rgba(255,255,255,0.2); padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;'>{st.session_state.user_profile['education_level']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Menu
    page = option_menu(
        None, 
        ["Chat", "Journal", "Mood Tracker", "Achievements", "Exercises", "Resources"],
        icons=['chat-dots-fill', 'pencil-square', 'graph-up-arrow', 'trophy-fill', 'activity', 'info-circle-fill'],
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color": "white"},
            "nav-link-selected": {"background-color": "rgba(255,255,255,0.2)"},
        }
    )
    
    # Stats Display
    user_stats = load_user_stats(STATS_FILE)
    st.markdown("---")
    st.markdown("**Your Progress:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 1.5rem;'>💬</div>
            <div style='font-size: 1.2rem; font-weight: bold;'>{user_stats.get('chat_messages', 0)}</div>
            <div style='font-size: 0.8rem; opacity: 0.9;'>Messages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 1.5rem;'>📊</div>
            <div style='font-size: 1.2rem; font-weight: bold;'>{user_stats.get('mood_logs', 0)}</div>
            <div style='font-size: 0.8rem; opacity: 0.9;'>Mood Logs</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Logout Button
    if st.button("Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Warning Banner
    st.markdown("""
    <div class='warning-banner'>
        <strong>Prototype Version</strong><br>
        Not a substitute for professional medical care
    </div>
    """, unsafe_allow_html=True)
    
    # Team Credits
    st.markdown("---")
    st.markdown("**Development Team:**")
    contributors = ["Prabhleen Kaur", "Prashant Kumar", "Priyanshu Pandey", "Nikita Saini", "Nidhi", "Palak Goyal"]
    for contributor in contributors:
        st.markdown(f"• {contributor}")

# CHAT PAGE
if page == "Chat":
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='font-size: 2.5rem; background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            How are you feeling today?
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize messages
    if "messages" not in st.session_state:
        profile = st.session_state.user_profile
        welcome_msg = f"""Hi {st.session_state.username}! 

I'm Medico, your personalized AI wellness companion. I can see you're a {profile['age_group']} {profile['education_level']} student at {st.session_state.college_info['college_name']}.

I'm here to support you with whatever is on your mind - whether it's {', '.join(profile['career_concerns'][:2]) if profile.get('career_concerns') else 'academic stress, career planning, or just daily challenges'}.

What would you like to talk about today?"""
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
    
    # Display current emotion if available
    if 'last_emotion' in st.session_state and st.session_state.last_emotion:
        emotion_data = st.session_state.last_emotion
        intensity_color = "#ff6b6b" if emotion_data['intensity'] > 0.5 else "#667eea"
        st.markdown(f"""
        <div class='emotion-badge' style='background: linear-gradient(45deg, {intensity_color}, #764ba2);'>
            Current mood: {emotion_data['primary_emotion'].title()} 
            (intensity: {emotion_data['intensity']:.1f}/1.0)
        </div>
        """, unsafe_allow_html=True)
    
    # Chat messages with enhanced styling
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"<div class='chat-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"<div class='chat-message'>{prompt}</div>", unsafe_allow_html=True)
        
        # Increment chat counter
        increment_stat(STATS_FILE, "chat_messages")
        
        # Check for crisis
        if detect_crisis(prompt):
            crisis_response = f"""
I'm deeply concerned about your safety. Your well-being is the most important thing right now.

Please reach out immediately:
• Emergency Services: 112
• Crisis Helpline: 9152987821  
• Campus Counselor: {st.session_state.college_info.get('counselor_phone', 'Contact your college')}

You are not alone, and help is available.
"""
            with st.chat_message("assistant"):
                st.error(crisis_response)
            st.session_state.messages.append({"role": "assistant", "content": crisis_response})
        else:
            # Generate personalized response
            with st.chat_message("assistant"):
                with st.spinner("Medico is thinking..."):
                    response, emotion_data = get_enhanced_gemini_response(
                        prompt, 
                        st.session_state.messages, 
                        st.session_state.college_info,
                        st.session_state.user_profile
                    )
                    st.markdown(f"<div class='chat-message'>{response}</div>", unsafe_allow_html=True)
                    st.session_state.last_emotion = emotion_data
                    
            st.session_state.messages.append({"role": "assistant", "content": response})

# MOOD TRACKER PAGE
elif page == "Mood Tracker":
    st.markdown("""
    <div class='mood-tracker'>
        <h1 style='text-align: center; color: white; margin-bottom: 2rem;'>Mood Tracker</h1>
        <p style='text-align: center; color: white; font-size: 1.1rem;'>Track your emotional well-being over time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mood rating
    mood_score = st.slider(
        "Rate your current mood", 
        1, 10, 5,
        help="1 = Very Down, 10 = Excellent"
    )
    
    # Mood context
    mood_context = st.text_area(
        "What's affecting your mood today? (optional)",
        placeholder="Share what's on your mind..."
    )
    
    if st.button("Log My Mood", use_container_width=True):
        mood_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "mood_score": mood_score,
            "context": mood_context,
            "user_profile": json.dumps(st.session_state.user_profile)
        }
        log_event(MOOD_LOG_FILE, mood_data)
        increment_stat(STATS_FILE, "mood_logs")
        
        # Personalized feedback based on score
        if mood_score >= 8:
            st.success("Wonderful! You're feeling great today. Keep up the positive energy!")
        elif mood_score >= 6:
            st.info("You're doing well! Remember to maintain this positive momentum.")
        elif mood_score >= 4:
            st.warning("You're managing okay. Consider some self-care activities today.")
        else:
            st.error("I'm sorry you're feeling down. Remember, it's okay to seek support when needed.")
    
    # Mood visualization
    if os.path.exists(MOOD_LOG_FILE):
        st.markdown("---")
        st.subheader("Your Mood Trends")
        
        mood_df = pd.read_csv(MOOD_LOG_FILE)
        mood_df['timestamp'] = pd.to_datetime(mood_df['timestamp'])
        
        if not mood_df.empty:
            # Create enhanced mood chart
            fig = px.line(
                mood_df, 
                x='timestamp', 
                y='mood_score', 
                markers=True,
                title="Your Mood Journey",
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#333',
                title_font_size=20
            )
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=8, line=dict(width=2, color='white'))
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mood insights
            avg_mood = mood_df['mood_score'].mean()
            recent_trend = mood_df.tail(5)['mood_score'].mean() - mood_df.head(5)['mood_score'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Mood", f"{avg_mood:.1f}/10")
            with col2:
                st.metric("Recent Trend", f"+{recent_trend:.1f}" if recent_trend > 0 else f"{recent_trend:.1f}")
            with col3:
                st.metric("Total Logs", len(mood_df))

# JOURNAL PAGE
elif page == "Journal":
    st.header("My Private Journal")
    st.write("A safe space for your thoughts and reflections")
    
    # Journal entry form
    journal_entry = st.text_area(
        "Write a new journal entry...", 
        height=300,
        placeholder="Express your thoughts freely... This is your private space."
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Save Entry", use_container_width=True):
            if journal_entry:
                entry_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "entry": journal_entry,
                    "word_count": len(journal_entry.split()),
                    "emotion_analysis": json.dumps(detect_emotion(journal_entry))
                }
                log_event(JOURNAL_FILE, entry_data)
                increment_stat(STATS_FILE, "journal_entries")
                st.success("Your journal entry has been saved securely!")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Please write something before saving.")
    
    # Display past entries
    if os.path.exists(JOURNAL_FILE):
        st.markdown("---")
        st.subheader("Past Entries")
        
        journal_df = pd.read_csv(JOURNAL_FILE)
        
        for index, row in reversed(list(journal_df.iterrows())):
            entry_date = pd.to_datetime(row['timestamp']).strftime('%B %d, %Y at %H:%M UTC')
            word_count = row.get('word_count', 'Unknown')
            
            with st.expander(f"Entry from {entry_date} ({word_count} words)"):
                st.write(row['entry'])
                
                # Show emotion analysis if available
                if 'emotion_analysis' in row and pd.notna(row['emotion_analysis']):
                    try:
                        emotion_data = json.loads(row['emotion_analysis'])
                        st.markdown(f"**Detected emotion:** {emotion_data['primary_emotion'].title()}")
                    except:
                        pass

# ACHIEVEMENTS PAGE
elif page == "Achievements":
    st.markdown("""
    <div style='text-align: center; margin-bottom: 3rem;'>
        <h1 style='font-size: 2.5rem; background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Your Achievements
        </h1>
        <p style='font-size: 1.2rem; color: #666;'>Celebrate your wellness journey! Every step matters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    user_stats = load_user_stats(STATS_FILE)
    
    # Achievement definitions with enhanced criteria
    achievements = {
        "First Steps": {
            "emoji": "👣", 
            "desc": "Started your wellness journey", 
            "unlocked": True,
            "category": "Getting Started"
        },
        "Conversationalist": {
            "emoji": "💬", 
            "desc": "Sent your first message", 
            "unlocked": user_stats.get("chat_messages", 0) >= 1,
            "category": "Communication"
        },
        "Deep Thinker": {
            "emoji": "🤔", 
            "desc": "Had 10+ conversations", 
            "unlocked": user_stats.get("chat_messages", 0) >= 10,
            "category": "Communication"
        },
        "Journal Starter": {
            "emoji": "✍️", 
            "desc": "Wrote your first journal entry", 
            "unlocked": user_stats.get("journal_entries", 0) >= 1,
            "category": "Reflection"
        },
        "Reflective Writer": {
            "emoji": "📖", 
            "desc": "Wrote 5+ journal entries", 
            "unlocked": user_stats.get("journal_entries", 0) >= 5,
            "category": "Reflection"
        },
        "Mood Tracker": {
            "emoji": "📊", 
            "desc": "Logged your first mood", 
            "unlocked": user_stats.get("mood_logs", 0) >= 1,
            "category": "Self-Awareness"
        },
        "Consistent Logger": {
            "emoji": "🗓️", 
            "desc": "Logged mood 7+ times", 
            "unlocked": user_stats.get("mood_logs", 0) >= 7,
            "category": "Self-Awareness"
        },
        "Wellness Champion": {
            "emoji": "🥇", 
            "desc": "Used all features actively", 
            "unlocked": (user_stats.get("mood_logs", 0) >= 5 and 
                        user_stats.get("journal_entries", 0) >= 3 and 
                        user_stats.get("chat_messages", 0) >= 20),
            "category": "Mastery"
        }
    }
    
    # Group achievements by category
    categories = {}
    for name, data in achievements.items():
        category = data['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((name, data))
    
    # Display achievements by category
    for category, items in categories.items():
        st.subheader(f"{category}")
        cols = st.columns(min(len(items), 4))
        
        for idx, (name, data) in enumerate(items):
            with cols[idx % 4]:
                status_class = "achievement-card" if data["unlocked"] else "achievement-card locked"
                opacity = "1" if data["unlocked"] else "0.5"
                
                st.markdown(f"""
                <div class='{status_class}' style='opacity: {opacity};'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>{data["emoji"]}</div>
                    <div style='font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;'>{name}</div>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>{data["desc"]}</div>
                    <div style='margin-top: 1rem; font-size: 0.8rem;'>
                        {'UNLOCKED' if data["unlocked"] else 'LOCKED'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Progress summary
    unlocked_count = sum(1 for achievement in achievements.values() if achievement["unlocked"])
    total_count = len(achievements)
    progress_percentage = (unlocked_count / total_count) * 100
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 20px; color: white;'>
        <h3>Overall Progress</h3>
        <div style='font-size: 2rem; margin: 1rem 0;'>{unlocked_count}/{total_count} Achievements</div>
        <div style='font-size: 1.2rem;'>{progress_percentage:.1f}% Complete</div>
    </div>
    """, unsafe_allow_html=True)

# GUIDED EXERCISES PAGE
elif page == "Exercises":
    st.header("Guided Wellness Exercises")
    st.write("Take a moment for yourself with these evidence-based wellness activities")
    
    # Exercise categories
    exercise_tabs = st.tabs(["Breathing", "Meditation", "Movement", "Mindfulness"])
    
    with exercise_tabs[0]:  # Breathing
        st.subheader("Breathing Exercises")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **4-7-8 Breathing Technique**
            1. Inhale for 4 counts
            2. Hold for 7 counts  
            3. Exhale for 8 counts
            4. Repeat 4 times
            
            *Great for anxiety and sleep*
            """)
            
        with col2:
            st.video("https://www.youtube.com/watch?v=inpok4MKVLM")
    
    with exercise_tabs[1]:  # Meditation
        st.subheader("Meditation Practices")
        
        st.markdown("""
        **5-Minute Body Scan**
        - Start with your toes
        - Slowly move attention up your body
        - Notice tension without judgment
        - Breathe into tight areas
        """)
        
        st.video("https://www.youtube.com/watch?v=O-6f5wQXSu8")
    
    with exercise_tabs[2]:  # Movement
        st.subheader("Gentle Movement") 
        
        st.markdown("""
        **Desk Stretches (Perfect for study breaks)**
        - Neck rolls (5 each direction)
        - Shoulder shrugs (10 times)
        - Seated spinal twist
        - Ankle circles
        
        *Set a timer to do these every hour!*
        """)
    
    with exercise_tabs[3]:  # Mindfulness
        st.subheader("Mindfulness Activities")
        
        st.markdown("""
        **5-4-3-2-1 Grounding Technique**
        - 5 things you can see
        - 4 things you can touch
        - 3 things you can hear
        - 2 things you can smell
        - 1 thing you can taste
        
        *Use when feeling overwhelmed*
        """)

# RESOURCES PAGE
elif page == "Resources":
    st.header("Wellness Resources & Support")
    st.write("Comprehensive resources tailored to your needs")
    
    # Personalized resources based on user profile
    profile = st.session_state.user_profile
    college_info = st.session_state.college_info
    
    # Emergency resources (always first)
    st.error(f"""
    **Emergency Resources**
    
    If you're in immediate danger:
    • **Call 112** (National Emergency)
    • **Call 9152987821** (Crisis Helpline)
    • Go to your nearest emergency room
    """)
    
    # College-specific resources
    st.info(f"""
    **{college_info['college_name']} Resources**
    
    • **Counselor:** {college_info.get('counselor_name', 'Contact student services')}
      Location: {college_info.get('counselor_location', 'Campus location')}
      Phone: {college_info.get('counselor_phone', 'Contact number')}
    
    • **Campus Doctor:** {college_info.get('doctor_name', 'Contact health center')}  
      Location: {college_info.get('doctor_location', 'Health center location')}
      Phone: {college_info.get('doctor_phone', 'Contact number')}
    """)
    
    # Personalized resources based on profile
    if profile.get('career_concerns'):
        st.success("""
        **Career & Academic Support**
        
        • **Study techniques** for exam preparation
        • **Interview preparation** guides and practice
        • **Career counseling** services
        • **Academic stress management** strategies
        """)
    
    # Resources based on demographics
    if profile.get('age_group') == '18-25':
        st.info("""
        **Young Adult Resources**
        
        • Transition to independence support
        • Peer support groups
        • Financial literacy resources
        • Social skills development
        """)
    
    # Additional helpful resources
    st.success("""
    **General Wellness Resources**
    
    • **Apps:** Headspace, Calm, Insight Timer
    • **Books:** "The Anxiety and Worry Workbook" by David Clark
    • **Websites:** Mental Health America, NAMI
    • **Hotlines:** Available 24/7 for support
    """)
    
    # Self-help techniques
    with st.expander("Self-Help Techniques"):
        st.markdown("""
        **Daily Wellness Practices:**
        
        1. **Morning routine** - Start with intention
        2. **Gratitude journaling** - Write 3 things daily  
        3. **Regular exercise** - Even 10 minutes helps
        4. **Healthy sleep** - 7-9 hours nightly
        5. **Social connection** - Reach out to others
        6. **Limit social media** - Take regular breaks
        7. **Practice mindfulness** - Stay present
        """)

# Footer
st.markdown("""
---
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Made with ❤️ by the Medico Team for your wellness journey</p>
    <p><em>Remember: This AI companion supplements but never replaces professional mental health care</em></p>
</div>
""", unsafe_allow_html=True)
