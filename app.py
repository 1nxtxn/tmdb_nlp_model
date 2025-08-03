import streamlit as st
from tmdb_nlp_model.model import MovieRecommender

@st.cache_resource
def get_recommender():
    # Instantiate once and cache
    return MovieRecommender()

rec = get_recommender()

st.title("🎬 TMDB Movie Recommender")

# 1) Get user inputs
user_text = st.text_input("What kind of movie are you in the mood for?")
genres  = ['Any'] + rec.print_genre_suggestions()
region  = rec.print_region_suggestions()

col1, col2 = st.columns(2)
with col1:
    genre      = st.selectbox("Genre", genres)
    movie_regn = st.selectbox("Region", region)
with col2:
    start_year = st.number_input("Start Year", 1900, 2025, 2000)
    end_year   = st.number_input("End Year",   1900, 2025, 2025)

# 2) When they click “Recommend”, show top-3
if st.button("Recommend"):
    if not user_text:
        st.warning("Please enter a description of the movie you’d like.")
    else:
        df_out = rec.recommend_movies(
            user_text, genre, start_year, end_year, movie_regn
        )
        if df_out.empty:
            st.info("No matches found—try relaxing your filters.")
        else:
            st.dataframe(df_out, use_container_width=True)
