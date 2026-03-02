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

# --- Page Config & Styling ---
st.set_page_config(page_title="AI Note Level-Up", page_icon="✨", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: white; }
    .stButton>button {
        background: linear-gradient(45deg, #6366f1, #a855f7);
        color: white;
        border: none;
        border-radius: 10px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Init OCR & API ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# Get API Key from Streamlit Secrets (for deployment) or Environment
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --- Logic Functions ---
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def extract_text(processed_img):
    try:
        result = reader.readtext(processed_img)
        text_list = [detection[1] for detection in result]
        return " ".join(text_list) if text_list else "❌ No text detected."
    except Exception as e:
        return f"OCR Error: {str(e)}"

def clean_notes(text):
    if not client:
        return "❌ API Key missing. Please add GROQ_API_KEY to secrets."
    try:
        prompt = f"Convert this messy OCR text into aesthetic, well-structured digital notes. Fix spelling, use bullet points, and add headers. Keep it real—don't add fake info.\n\nTEXT:\n{text}"
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {str(e)}"

# --- Export Logic (Memory-based) ---
def get_docx(text):
    doc = Document()
    for line in text.split("\n"):
        if line.strip():
            p = doc.add_paragraph(line)
            if len(line) < 60 and not line.startswith("-"):
                p.runs[0].bold = True
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def get_pdf(text):
    bio = io.BytesIO()
    canv = canvas.Canvas(bio, pagesize=letter)
    y = 750
    canv.setFont("Helvetica", 10)
    for line in text.split("\n"):
        if y < 50:
            canv.showPage()
            y = 750
        canv.drawString(50, y, line[:90])
        y -= 15
    canv.save()
    return bio.getvalue()

# --- Streamlit UI ---
st.title("✨ AI Note Level-Up")
st.write("Turn your messy scribbles into clean digital notes instantly.")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("📸 Upload Scribbles", type=["png", "jpg", "jpeg"])
    if uploaded_file and st.button("Clean My Notes ✨"):
        with st.spinner("Processing..."):
            # Process
            processed_img = preprocess_image(uploaded_file)
            raw_text = extract_text(processed_img)
            cleaned_text = clean_notes(raw_text)
            
            # Save to session state to prevent refresh loss
            st.session_state['raw'] = raw_text
            st.session_state['clean'] = cleaned_text

if 'clean' in st.session_state:
    with col2:
        tab1, tab2 = st.tabs(["Cleaned Notes", "Raw OCR"])
        
        with tab1:
            st.text_area("Digital Version", st.session_state['clean'], height=300)
            
            # Downloads
            c1, c2 = st.columns(2)
            docx_data = get_docx(st.session_state['clean'])
            pdf_data = get_pdf(st.session_state['clean'])
            
            c1.download_button("📥 Download .docx", data=docx_data, file_name="Notes.docx")
            c2.download_button("📥 Download .pdf", data=pdf_data, file_name="Notes.pdf")
            
        with tab2:
            st.text_area("Initial Scan", st.session_state['raw'], height=300)
