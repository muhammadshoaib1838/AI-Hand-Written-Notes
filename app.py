import streamlit as st
import easyocr
import cv2
import numpy as np
from groq import Groq
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import os

# --- Page Config ---
st.set_page_config(page_title="AI Note Pro", page_icon="📝", layout="wide")

# --- Professional Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #f1f5f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        background-color: #1e293b; 
        border-radius: 10px 10px 0 0; 
        color: white; 
    }
    .stTabs [aria-selected="true"] { background-color: #6366f1 !important; }
    
    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px;
        width: 100%;
    }
    
    .signature { color: #818cf8; text-align: center; margin-top: 50px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Init OCR & API ---
@st.cache_resource
def load_ocr():
    # gpu=False is required for Streamlit Cloud Free Tier
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --- Logic Functions ---
def process_pipeline(uploaded_file):
    try:
        # 1. OCR Stage
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        if img is None:
            return None, None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (5, 5), 0),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        raw_text = " ".join(reader.readtext(processed_img, detail=0))
        
        if not raw_text.strip():
            return "No text detected", None, None

        # 2. AI Structuring Stage
        if not client:
            return raw_text, "Error: API Key missing", "N/A Bird"

        prompt = (
            "Act as a Professional Document Architect. "
            "Convert this OCR text into a beautiful digital document: "
            "- Use # for the Main Title "
            "- Use
