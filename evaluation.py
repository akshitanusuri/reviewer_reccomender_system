import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer, util
import torch
import matplotlib.pyplot as plt

# -------------------------------
# Load author dataset
# -------------------------------
with open("authors.json", "r", encoding="utf-8") as f:
    authors = json.load(f)

author_names = list(authors.keys())
author_texts = list(authors.values())

# -------------------------------
# Load one sample paper as input (you can change this)
# -------------------------------
from utils import extract_text_from_pdf
input_text = extract_text_from_pdf("temp.pdf")  # test paper

# -------------------------------
# 1️⃣ TF-IDF Similarity
# -------------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(author_texts + [input_text])
similarity_tfidf = (tfidf_matrix[-1] @ tfidf_matrix[:-1].T).toarray()[0]

# -------------------------------
# 2️⃣ Doc2Vec Similarity
# -------------------------------
tagged_docs = [TaggedDocument(words=text.split(), tags=[i]) for i, text in enumerate(author_texts)]
doc2vec_model = Doc2Vec(tagged_docs, vector_size=128, window=5, min_count=2, epochs=20)
input_vec = doc2vec_model.infer_vector(input_text.split())
doc2vec_vectors = np.array([doc2vec_model.dv[i] for i in range(len(author_texts))])
similarity_doc2vec = np.dot(doc2vec_vectors, input_vec) / (
    np.linalg.norm(doc2vec_vectors, axis=1) * np.linalg.norm(input_vec)
)

# -------------------------------
# 3️⃣ SentenceTransformer Similarity
# -------------------------------
st_model = SentenceTransformer('all-MiniLM-L6-v2')
author_embeddings = st_model.encode(author_texts, convert_to_tensor=True)
input_embedding = st_model.encode(input_text, convert_to_tensor=True)
similarity_st = util.pytorch_cos_sim(input_embedding, author_embeddings)[0].cpu().numpy()

# -------------------------------
# Combine & Display
# -------------------------------
methods = ["TF-IDF", "Doc2Vec", "SentenceTransformer"]
avg_scores = [
    np.mean(similarity_tfidf),
    np.mean(similarity_doc2vec),
    np.mean(similarity_st),
]

print("\n🔍 Average Similarity Scores:")
for name, score in zip(methods, avg_scores):
    print(f"{name:<25}: {score:.4f}")

# Top reviewer by each method
print("\n🏆 Top Reviewers:")
print(f"TF-IDF              → {author_names[np.argmax(similarity_tfidf)]}")
print(f"Doc2Vec             → {author_names[np.argmax(similarity_doc2vec)]}")
print(f"SentenceTransformer → {author_names[np.argmax(similarity_st)]}")

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(8, 5))
plt.bar(methods, avg_scores)
plt.title("Average Similarity Scores by Method")
plt.ylabel("Mean Similarity")
plt.xlabel("Method")
plt.show()
