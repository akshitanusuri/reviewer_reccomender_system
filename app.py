import streamlit as st
from utils import extract_text_from_pdf
from reviewer_engine import get_top_k_reviewers
import json

st.title("📑 Reviewer Recommendation System")
st.caption("Find top researchers best suited to review your paper")

# Load authors (optional: for warnings or checks)
try:
    with open("authors.json", "r", encoding="utf-8") as f:
        authors = json.load(f)
except Exception as e:
    st.error(f"Error loading authors.json: {e}")
    authors = {}

# Check if authors dataset is valid
if not authors or all(not text.strip() for text in authors.values()):
    st.warning("⚠️ No valid authors found. Check your authors dataset.")

uploaded_file = st.file_uploader("Upload your research paper (PDF)", type="pdf")
k = st.slider("Select number of reviewers to suggest", 1, 10, 3)  # default top 3

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded successfully!")

    if st.button("Find Top Reviewers"):
        try:
            input_text = extract_text_from_pdf("temp.pdf")
            st.info("Analyzing paper and finding suitable reviewers...")

            # Use precomputed embeddings in reviewer_engine.py
            top_reviewers = get_top_k_reviewers(input_text, k, embeddings_path="author_embeddings.json")

            if not top_reviewers:
                st.warning("⚠️ No reviewers found. Check your authors dataset.")
            else:
                st.subheader("🏆 Top Reviewers")
                for i, (name, score) in enumerate(top_reviewers):
                    display_score = max(score, 0.0)
                    if i == 0:
                        st.write(f"🏆 **{name}** — Similarity Score: {display_score:.4f}")
                    else:
                        st.write(f"{name} — Similarity Score: {display_score:.4f}")
        except Exception as e:
            st.error(f"Error fetching top reviewers: {e}")
