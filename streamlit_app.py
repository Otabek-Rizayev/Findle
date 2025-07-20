import os
import pickle
import pathlib
import platform
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

@st.cache_resource
def load_data():
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return data["dataset"], data["embeddings"], model

profiles, profile_embeddings, model = load_data()
st.title("Findle AI")
st.write(
    "Describe your needed Person!"
)
user_text = st.text_area("ABOUT", "QWE")

if st.button("DEEP RESEARCH 🔭"):
    if not user_text.strip():
        st.warning("Please, Describe your needed Person!")
    else:
        user_embedding = model.encode(user_text, convert_to_numpy=True)
        cos_scores = np.dot(profile_embeddings, user_embedding) / (
            np.linalg.norm(profile_embeddings, axis=1) * np.linalg.norm(user_embedding)
        )
        top_idx = np.argsort(-cos_scores)[:5]
        st.markdown("### Top 5 Similar Profiles:")
        for idx in top_idx: 
            st.markdown("---")
            st.markdown(f"**Similarity Accuracy:** {cos_scores[idx]:.2f}")
            st.markdown(f"**NAME:** {profiles[idx]['FirstName']} {profiles[idx]['LastName']}")
            st.markdown(f"**ABOUT:** {profiles[idx]['About Me']}")
            st.markdown(f"**EXPERIENCE:** {profiles[idx]['Experience']}")
            st.markdown(f"**SKILLS:** {profiles[idx]['Skills']}")
            st.markdown(f"**HEADLINE:** {profiles[idx]['Headline']}")

