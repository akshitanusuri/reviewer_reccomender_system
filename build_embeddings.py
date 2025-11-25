import os
import json
from sentence_transformers import SentenceTransformer
from utils import extract_text_from_pdf

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_author_embeddings(
    data_dir="data/authors_papers",
    corpus_path="authors.json",
    embeddings_path="author_embeddings.json"
):
    authors = {}
    embeddings = {}

    # Build author corpus
    for author_folder in os.listdir(data_dir):
        author_path = os.path.join(data_dir, author_folder)
        if os.path.isdir(author_path):
            all_text = ""
            pdfs_found = 0
            for pdf in os.listdir(author_path):
                if pdf.endswith(".pdf"):
                    pdf_path = os.path.join(author_path, pdf)
                    if not os.path.exists(pdf_path):
                        print(f"Warning: PDF not found: {pdf_path}")
                        continue
                    try:
                        text = extract_text_from_pdf(pdf_path)
                        if text.strip():
                            all_text += text + " "
                            pdfs_found += 1
                        else:
                            print(f"Warning: PDF has no extractable text: {pdf_path}")
                    except Exception as e:
                        print(f"Error reading PDF {pdf_path}: {e}")
            if all_text.strip():
                authors[author_folder] = all_text.strip()
                print(f"Processed author: {author_folder}, PDFs used: {pdfs_found}")
            else:
                print(f"No valid text found for author: {author_folder}, skipping.")

    if not authors:
        print("No authors with valid PDFs found. Check your data directory!")
        return

    # Save corpus
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(authors, f, ensure_ascii=False, indent=2)
    print(f"Saved author corpus to {corpus_path}")

    # Compute embeddings
    author_names = list(authors.keys())
    author_texts = list(authors.values())
    author_embeddings = model.encode(author_texts, convert_to_tensor=True)

    # Save embeddings as list (tensor -> list)
    for i, name in enumerate(author_names):
        embeddings[name] = author_embeddings[i].cpu().tolist()

    with open(embeddings_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
    print(f"Saved author embeddings to {embeddings_path}")

if __name__ == "__main__":
    build_author_embeddings()
