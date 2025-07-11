import streamlit as st
from pypdf import PdfReader
from transformers import pipeline
from reportlab.pdfgen import canvas
import tempfile
import os
from transformers import AutoTokenizer

os.environ['CURL_CA_BUNDLE'] = ''  # Disable SSL certificate verification for Hugging Face requests

st.set_page_config(page_title="PDF Summarizer", layout="centered")
st.title("PDF Summarizer App")

st.markdown("""
Upload a PDF file and get a summary using a free, open-source language model (facebook/bart-large-cnn).
You can also generate a sample PDF for testing.
""")

# Generate sample PDF
def generate_sample_pdf():
    temp_dir = tempfile.gettempdir()
    sample_path = os.path.join(temp_dir, "sample.pdf")
    c = canvas.Canvas(sample_path)
    c.drawString(100, 750, "This is a sample PDF file for testing PDF summarization.")
    c.drawString(100, 730, "You can upload your own PDF to get a summary.")
    c.save()
    return sample_path

if st.button("Generate Sample PDF"):
    sample_path = generate_sample_pdf()
    with open(sample_path, "rb") as f:
        st.download_button("Download Sample PDF", f, file_name="sample.pdf")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
pdf_text = ""
if uploaded_file:
    reader = PdfReader(uploaded_file)
    pdf_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    st.text_area("Extracted PDF Text (first 1000 chars)", pdf_text[:1000], height=200)

    if st.button("Summarize PDF"):
        with st.spinner("Loading summarization model and generating summary..."):
            model_name = "facebook/bart-large-cnn"  # or "google/pegasus-xsum"
            summarizer = pipeline(
                "summarization",
                model=model_name
            )
            # Truncate raw text to 1024 tokens
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokens = tokenizer(pdf_text, truncation=True, max_length=1024)
            truncated_text = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
            summary = summarizer(truncated_text, max_length=200, min_length=30, do_sample=False)[0]['summary_text']
        st.subheader("Summary")
        st.write(summary)
else:
    st.info("Upload a PDF file to get started or generate a sample PDF above.")
