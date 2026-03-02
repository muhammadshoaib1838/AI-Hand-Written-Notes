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
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1e293b; border-radius: 10px 10px 0 0; color: white; padding: 10px; }
    .stTabs [aria-selected="true"] { background-color: #6366f1 !important; }
    .note-card {
        background-color: #1e293b;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #818cf8;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    .highlight { color: #f472b6; font-weight: bold; }
    .header-text { color: #818cf8; font-weight: 800; font-size: 1.5rem; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Init OCR & API ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --- Logic Functions ---
def preprocess_image(uploaded_file):
    # FIXED: Added 'np.' before uint8
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def extract_text(processed_img):
    result = reader.readtext(processed_img)
    return " ".join([detection[1] for detection in result]) if result else ""

def clean_notes(text):
    if not client: return "❌ Missing Groq API Key."
    prompt = (
        "Convert this OCR text into a professional, aesthetic document. "
        "Use Markdown: Use '###' for colorful headers, bold key terms, and bullet points. "
        "Fix all grammar. Format it like high-quality digital study notes.\n\n"
        f"TEXT: {text}"
    )
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    return completion.choices[0].message.content

# --- Export Helpers ---
def get_docx(text):
    doc = Document()
    doc.add_heading('Digitized Notes', 0)
    for line in text.split("\n"):
        if line.strip(): doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# --- UI Layout ---
st.title("✨ AI Note Level-Up")
st.caption("Transform handwritten scribbles into professional digital assets.")

col_input, col_output = st.columns([1, 1.5])

with col_input:
    st.subheader("📤 Source Material")
    file = st.file_uploader("Upload image (PNG, JPG)", type=["png", "jpg", "jpeg"])
    
    if file:
        st.image(file, caption="Original Scribbles", use_container_width=True)
        if st.button("🚀 Process & Polish", use_container_width=True):
            with st.spinner("Analyzing handwriting..."):
                raw = extract_text(preprocess_image(file))
                if raw:
                    st.session_state['clean'] = clean_notes(raw)
                    st.session_state['raw'] = raw
                else:
                    st.error("Could not find any text in that image!")

with col_output:
    if 'clean' in st.session_state:
        st.subheader("💎 Polished Results")
        
        tab_view, tab_raw = st.tabs(["✨ Digital Note", "📄 Raw Data"])
        
        with tab_view:
            # Displaying with markdown for colors and bolding
            st.markdown(
