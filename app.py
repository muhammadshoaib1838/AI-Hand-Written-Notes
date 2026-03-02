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
    }
    
    .signature { color: #818cf8; text-align: center; margin-top: 50px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Init OCR & API ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

# Initialize client only if key exists
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --- Logic Functions ---
def preprocess_image(uploaded_file):
    # Fixed: Ensure file pointer is at start
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None
    # Standard grayscale for OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def extract_text(img):
    result = reader.readtext(img)
    return " ".join([detection[1] for detection in result]) if result else ""

def clean_notes(text):
    if not client: return "❌ Missing Groq API Key in Secrets."
    prompt = (
        "Convert this OCR text into a professional document. "
        "1. Use <h2 style='color:#818cf8;'> for headings. "
        "2. Use <span style='color:#f472b6; font-weight:bold;'> for key terms. "
        "3. Fix all grammar and format as high-quality study notes. "
        "End with '--- END ---'."
    )
    completion =
