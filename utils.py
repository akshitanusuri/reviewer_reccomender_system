import fitz
import re
import os

def extract_text_from_pdf(pdf_path):
    """
    Extract raw text from a PDF file.
    Returns empty string if file is missing or cannot be read.
    """
    if not os.path.exists(pdf_path):
        print(f"Warning: PDF not found: {pdf_path}")
        return ""

    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        if not text.strip():
            print(f"Warning: No extractable text in PDF: {pdf_path}")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""
