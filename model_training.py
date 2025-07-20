import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocessing Datasets
dataset = load_dataset("ilsilfverskiold/linkedin_profiles_synthetic", split="train")
df = dataset.to_pandas()
def profile_to_text(example):
    return f"{example['About Me']} {example['Experience']} {example['Skills']} {example['Headline']}"

profile_texts = [profile_to_text(row) for row in dataset]
print(type(dataset))

# Datasets Embeddings
model = SentenceTransformer("all-mpnet-base-v2")
profile_embeddings = model.encode(
    profile_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=32,
    show_progress_bar=True
).astype("float32")

# Test Semantich Search
query = "AI enginner from Uzbekistan, 1 year experience and working on Fraud Detector project"
query_embedding = model.encode(query, normalize_embeddings=True).astype("float32").reshape(1, -1)
similarities = cosine_similarity(query_embedding, profile_embeddings)[0]
top_k = 5
top_indices = np.argsort(similarities)[::-1][:top_k]
print("\nTop", top_k, "Similar Profiles:\n")
for idx in top_indices:
    i = int(idx)  # <-- MUHIM O'ZGARISH
    profile = dataset[i]  # <-- Endi xatolik yo‘q
    score = similarities[i]
    print(f"🔹 Similarity: {score:.3f}")
    print(f"👤 Name: {profile.get('FirstName', '')} {profile.get('LastName', '')}")
    print(f"💼 Headline: {profile.get('Headline', '')}")
    print(f"📍 Location: {profile.get('Location', '')}")
    print(f"🧠 Skills: {profile.get('Skills', '')}")
    print(f"💼 Experience: {profile.get('Experience', '')}")
    print(f"📄 About: {profile.get('About Me', '')[:200]}...")
    print("—" * 60)

# Save Model
import pickle
fields = ["FirstName", "LastName", "Location", "About Me", "Experience", "Skills", "Headline"]
profiles = [{field: str(row[field]) if field in row else "" for field in fields} for _, row in df.iterrows()]
with open("model.pkl", "wb") as f:
    pickle.dump({"dataset": profiles, "embeddings": profile_embeddings}, f)