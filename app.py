import streamlit as st
from transformers import pipeline
import os
import platform

# Set tesseract path only on Windows
# if platform.system() == "Windows":
#     pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load models
@st.cache_resource
def load_models():
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return sentiment_pipeline

sentiment_model = load_models()

# Theme toggle using session state
if 'theme_dark' not in st.session_state:
    st.session_state.theme_dark = True

def toggle_theme():
    st.session_state.theme_dark = not st.session_state.theme_dark

# UI
theme_icon = "🌑" if st.session_state.theme_dark else "🌕"
st.markdown(f"""
    <div style='text-align: right'>
        <button onclick="window.location.reload()" style="background:none;border:none;font-size:24px;">{theme_icon}</button>
    </div>
""", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown(f"""
    <style>
        html, body {{
            background-color: {"#1e1e1e" if st.session_state.theme_dark else "#f0f8ff"};
            color: {"#ffffff" if st.session_state.theme_dark else "#000000"};
            transition: background-color 0.4s ease;
        }}
        textarea, .stTextInput > div > div > input {{
            background-color: {"#333333" if st.session_state.theme_dark else "#ffffff"} !important;
            color: {"#ffffff" if st.session_state.theme_dark else "#000000"} !important;
        }}
        .stButton button {{
            background: linear-gradient(135deg, #ff7e5f, #feb47b);
            color: white;
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            transition: transform 0.2s ease, background 0.3s ease;
            box-shadow: 0 6px 16px rgba(0,0,0,0.3);
        }}
        .stButton button:hover {{
            background: linear-gradient(135deg, #feb47b, #ff7e5f);
            transform: scale(1.05);
        }}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Sentiment Analyzer")
st.markdown("Analyze text for **sentiment** using AI models.")

# Text input
text_input = st.text_area("Enter text to analyze:")

if st.button("Analyze Text"):
    if text_input:
        with st.spinner("Analyzing..."):
            sentiment_result = sentiment_model(text_input)[0]
            sentiment_emoji = "😊" if sentiment_result['label'] == "POSITIVE" else "😞"

            st.subheader("🔍 Text Result:")
            st.markdown(f"**Sentiment:** {sentiment_result['label']} {sentiment_emoji} ({sentiment_result['score']:.2f})")
    else:
        st.warning("Please enter some text.")
