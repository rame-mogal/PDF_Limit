import os
import json
import re
import fitz  # PyMuPDF
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]

#Streamlit UI
st.title("Boltware PDF Extractor 🔍")
st.write("Upload a PDF. Tesseract OCR will run only if the PDF is scanned (image-based).")

uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

# Query OpenAI API
def query_openai_json(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return None

# GPT Prompt Builder
def build_prompt(text):
    return f"""
Extract the following fields from the text:

- Firm Name
- Address
- Date
- Turnover
- Contact Details

Text:
{text}

Respond ONLY in this JSON format:
{{
  "Firm Name": "...",
  "Address": "...",
  "Date": "...",
  "Turnover": "...",
  "Contact Details": "..."
}}
"""

# Tesseract OCR function
def tesseract_ocr_text(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)

# Detect if a page is image-based (scanned)
def is_scanned_page(page):
    text = page.get_text().strip()
    return len(text) < 20  # Consider scanned if minimal embedded text

# Main logic
if uploaded_file:
    with st.spinner("🔍 Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name

        doc = fitz.open(tmp_pdf_path)
        max_pages = min(2, len(doc))

        full_text = ""

        for i in range(max_pages):
            page = doc[i]
            if is_scanned_page(page):
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                try:
                    ocr_text = tesseract_ocr_text(img)
                    full_text += ocr_text + "\n"
                except Exception as e:
                    st.warning(f"OCR failed on page {i+1}: {e}")
            else:
                full_text += page.get_text() + "\n"

        # Chunk & process with GPT
        max_len = 2000
        chunks = [full_text[i:i + max_len] for i in range(0, len(full_text), max_len)]

        all_results = []

        for chunk in chunks[:1]:  # Only first chunk for now
            prompt = build_prompt(chunk)
            response = query_openai_json(prompt)
            if response:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        all_results.append(data)
                    except json.JSONDecodeError:
                        st.warning("⚠️ Invalid JSON format from OpenAI.")

        # Merge results
        final_result = {}
        for result in all_results:
            for key, value in result.items():
                if key not in final_result or not final_result[key]:
                    final_result[key] = value

        if final_result:
            df = pd.DataFrame([final_result])
            st.subheader("✅ Extracted Information")
            st.dataframe(df)

            st.download_button("⬇️ Download as JSON", json.dumps(final_result, indent=2), file_name="extracted_info.json")
            st.download_button("⬇️ Download as CSV", df.to_csv(index=False), file_name="extracted_info.csv")
        else:
            st.error("❌ No valid data extracted.")
