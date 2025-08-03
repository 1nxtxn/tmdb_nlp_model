# app.py
import os
import ast
import streamlit as st
import pandas as pd
import joblib
from scipy import sparse

# 1) Load metadata CSV & rehydrate list columns
@st.cache(allow_output_mutation=True)
def load_metadata(path="artifacts/movies_metadata.csv"):
    df = pd.read_csv(path)
    # parse back the genre_list and country_list
    df['genre_list']  = df['genre_list'].apply(ast.literal_eval)
    df['country_list']= df['country_list'].apply(ast.literal_eval)
    return df

# 2) Load vectorizer & TF-IDF matrix
@st.cache(allow_output_mutation=True)
def load_models(vec_path="artifacts/tfidf_vectorizer.joblib",
                mat_path="artifacts/tfidf_matrix.npz"):
    vec = joblib.load(vec_path)
    X   = sparse.load_npz(mat_path)
    return vec, X

# Actually load
df               = load_metadata()
vectorizer, X    = load_models()

# now import ONLY the pure logic
from model import recommend_movies, REGION_GROUPS

# --- Streamlit UI ---

st.title("🎬 AI4ALL Movie Recommender")

# genre picker
genres = sorted({g for sub in df['genre_list'] for g in sub})
genre  = st.sidebar.selectbox("Genre", ["any"] + genres)

# year range
min_y, max_y = int(df.release_year.min()), int(df.release_year.max())
start_y, end_y = st.sidebar.slider(
    "Release Year", min_y, max_y, (2000, 2020)
)

# region picker
regions = list(REGION_GROUPS.keys()) + ["any"]
region  = st.sidebar.selectbox("Region", regions)

# free-text
user_text = st.sidebar.text_area(
    "Describe what you like:",
    placeholder="e.g. emotional stakes, witty dialogue…"
)

if st.sidebar.button("Recommend"):
    if not user_text.strip():
        st.sidebar.error("Please enter some text.")
    else:
        recs = recommend_movies(
            user_text=user_text,
            user_genre=genre,
            start_year=start_y,
            end_year=end_y,
            movie_region=region,
            df=df,
            vectorizer=vectorizer,
            X=X,
            top_n=10
        )
        st.subheader(f"Top {len(recs)} picks for {genre} ({start_y}–{end_y}) in {region}")
        st.dataframe(recs)
