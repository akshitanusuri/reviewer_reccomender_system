import json
import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_top_k_reviewers(input_text, k=5, embeddings_path="author_embeddings.json"):
    # Load precomputed embeddings
    try:
        with open(embeddings_path, "r", encoding="utf-8") as f:
            embeddings_data = json.load(f)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return []

    if not embeddings_data:
        return []

    author_names = []
    author_embeddings = []

    # Convert JSON lists to float tensors
    for name, emb_list in embeddings_data.items():
        try:
            emb_floats = [float(x) for x in emb_list]  # ensure numeric
            author_embeddings.append(torch.tensor(emb_floats))
            author_names.append(name)
        except Exception as e:
            print(f"Skipping {name}, cannot convert embedding: {e}")

    if not author_embeddings:
        print("No valid embeddings found.")
        return []

    # Encode input paper
    input_embedding = model.encode(input_text, convert_to_tensor=True)

    # Compute similarity
    similarity = util.pytorch_cos_sim(input_embedding, torch.stack(author_embeddings))[0]

    # Pair names with similarity scores and sort top k
    pairs = [(author_names[i], float(similarity[i])) for i in range(len(author_names))]
    pairs.sort(key=lambda x: x[1], reverse=True)

    return pairs[:k]
