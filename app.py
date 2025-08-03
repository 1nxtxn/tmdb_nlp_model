import streamlit as st
import pandas as pd
from model import load_data, build_corpus_and_vectorizer, recommend_movies

@st.cache(allow_output_mutation=True)
def load_artifacts():
    # If you haven't generated artifacts yet, run `python model.py` first.
    vec = joblib.load('tfidf_vectorizer.joblib')
    X   = sparse.load_npz('tfidf_matrix.npz')
    df  = pd.read_csv('movies_metadata.csv')
    return df, vec, X

# Load everything once
df, vectorizer, tfidf_matrix = load_artifacts()

st.title("🎬 AI4ALL Movie Recommender")

# Sidebar Inputs
genres = sorted({g for sub in df['genre_list'] for g in sub})
genre = st.sidebar.selectbox("Genre", ["any"] + genres)

min_year, max_year = int(df.release_year.min()), int(df.release_year.max())
start_year, end_year = st.sidebar.slider(
    "Release Year Range", min_year, max_year, (2000, 2020)
)

regions = list(REGION_GROUPS.keys()) + ["any"]
region = st.sidebar.selectbox("Region", regions)

user_text = st.sidebar.text_area(
    "Describe what you like about past favorites:",
    placeholder="e.g. emotional stakes, strong performances..."
)

# Recommendation Trigger
if st.sidebar.button("Recommend"):
    if not user_text.strip():
        st.sidebar.error("Please enter a description.")
    else:
        recs = recommend_movies(
            user_text, genre, start_year, end_year, region,
            df, vectorizer, tfidf_matrix, top_n=10
        )
        st.subheader(f"Top {len(recs)} picks for '{genre}' ({start_year}-{end_year}) in {region}")
        st.dataframe(recs)
