import os
import json
import re
import fitz  # PyMuPDF
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import openai
import numpy as np
from paddleocr import PaddleOCR

# Load environment variables
load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]
# Initialize PaddleOCR (CPU version)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Streamlit UI
st.title("Boltware PDF Extractor 🔍")
st.write("Upload a scanned PDF. PaddleOCR + OpenAI GPT-3.5 Turbo will extract structured data from the first 2 pages.")

uploaded_file = st.file_uploader("📄 Upload your scanned PDF", type=["pdf"])

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

# Build prompt for GPT
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

# OCR function using PaddleOCR
def paddle_ocr_text(image: Image.Image):
    img_array = image.convert("RGB")
    results = ocr_model.ocr(np.array(img_array), cls=True)
    full_text = ""
    for line in results[0]:
        text = line[1][0]
        full_text += text + "\n"
    return full_text

# Main logic
if uploaded_file:
    with st.spinner("🔍 Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name

        doc = fitz.open(tmp_pdf_path)
        max_pages = min(2, len(doc))  # Limit to first 2 pages

        images = [
            Image.frombytes(
                "RGB",
                [page.get_pixmap(dpi=150).width, page.get_pixmap(dpi=150).height],
                page.get_pixmap(dpi=150).samples
            )
            for page in doc[:max_pages]
        ]

        ocr_texts = []
        for img in images:
            try:
                ocr_texts.append(paddle_ocr_text(img))
            except Exception as e:
                st.warning(f"OCR failed for one page: {e}")

        full_text = "\n".join(ocr_texts)
        max_len = 2000
        chunks = [full_text[i:i + max_len] for i in range(0, len(full_text), max_len)]

        all_results = []

        for chunk in chunks[:1]:  # Process only the first chunk for now
            prompt = build_prompt(chunk)
            response = query_openai_json(prompt)
            if response:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        all_results.append(data)
                    except json.JSONDecodeError:
                        st.warning("⚠️ Invalid JSON in response.")

        # Merge all extracted results
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
