# tmdb_nlp_model/model.py

import ast
import numpy as np
import pandas as pd
import nltk
import kagglehub
from kagglehub import KaggleDatasetAdapter

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

# Constants
SMALL_THRESHOLD = 20_000_000  # $20M cutoff
REGION_GROUPS = {
    "North America": ["United States", "Canada", "Mexico"],
    "South America": ["Brazil", "Argentina", "Colombia", "Chile", "Peru", "Venezuela", "Ecuador", "Bolivia", "Paraguay", "Uruguay"],
    "Europe": ["United Kingdom", "France", "Germany", "Italy", "Spain", "Netherlands", "Sweden", "Norway", "Denmark", "Finland", "Belgium", "Switzerland", "Austria", "Poland", "Russia"],
    "Asia": ["Japan", "China", "India", "South Korea", "Hong Kong", "Taiwan", "Thailand", "Singapore", "Malaysia", "Indonesia"],
    "Africa": ["Nigeria", "Egypt", "South Africa", "Kenya", "Morocco", "Algeria", "Ghana", "Ethiopia"],
    "Oceania": ["Australia", "New Zealand", "Fiji"],
    "Latin America": ["Brazil", "Argentina", "Colombia", "Mexico", "Chile", "Peru", "Venezuela", "Ecuador"]
}

class MovieRecommender:
    def __init__(
        self,
        dataset_name: str = "tmdb/tmdb-movie-metadata",
        file_path: str = "tmdb_5000_movies.csv",
        k_range: range = range(45, 54),
        tfidf_kwargs: dict = None
    ):
        # 1. Load data directly from Kaggle
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            dataset_name,
            file_path
        )

        # 2. Feature selection & cleaning
        df.drop_duplicates(subset=['original_title'], keep='first', inplace=True)
        df['revenue'] = df.get('revenue', 0).fillna(0).astype(float)
        df = df[[
            'id', 'original_title', 'overview', 'genres', 'keywords',
            'production_countries', 'release_date', 'vote_average', 'revenue'
        ]]
        df.dropna(subset=['overview'], inplace=True)
        df['keywords'] = df['keywords'].fillna('[]')
        df.reset_index(drop=True, inplace=True)

        # 3. Text preprocessing
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        def preprocess(tokens):
            tokens = [w.lower() for w in tokens if w.isalpha() or any(ch.isdigit() for ch in w)]
            tokens = [t for t in tokens if t not in stop_words]
            return [stemmer.stem(t) for t in tokens]

        df['tokenized_overview'] = df['overview'].apply(word_tokenize)
        df['clean_overview'] = df['tokenized_overview'].apply(preprocess).str.join(' ')
        def extract_keywords(kw_list):
            try:
                items = ast.literal_eval(kw_list)
                return ' '.join([i['name'] for i in items])
            except:
                return ''
        df['keywords_text'] = df['keywords'].apply(extract_keywords)
        df['text_input'] = (
            df['original_title'] + ' ' +
            df['keywords_text'] + ' ' +
            df['clean_overview']
        )

        # 4. TF-IDF vectorization
        tfidf_defaults = dict(ngram_range=(1,3), min_df=3, max_df=0.8,
                              sublinear_tf=True, max_features=30000)
        self.vectorizer = TfidfVectorizer(**(tfidf_kwargs or tfidf_defaults))
        X = self.vectorizer.fit_transform(df['text_input'])

        # 5. Find best k by silhouette
        sil_scores = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            sil_scores.append(silhouette_score(X, km.labels_))
        best_k = k_range[int(np.argmax(sil_scores))]

        # 6. Final clustering
        self.kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)
        df['cluster'] = self.kmeans.labels_

        # 7. Save to instance
        self.df = df
        self.X = X

    def show_example_matches(self, user_text: str, top_n: int = 5) -> pd.DataFrame:
        user_vec = self.vectorizer.transform([user_text])
        sims = cosine_similarity(user_vec, self.X).flatten()
        preview = self.df.copy()
        preview['similarity'] = sims
        preview['release_year'] = pd.to_datetime(
            preview['release_date'], errors='coerce'
        ).dt.year.fillna(0).astype(int)
        return preview.sort_values('similarity', ascending=False).head(top_n)[
            ['original_title', 'release_year', 'similarity']
        ]

    def print_genre_suggestions(self) -> list:
        all_genres = []
        for row in self.df['genres']:
            try:
                items = ast.literal_eval(row)
                all_genres += [g['name'] for g in items]
            except:
                continue
        return list(pd.Series(all_genres).value_counts().head(10).index)

    def print_region_suggestions(self) -> list:
        return list(REGION_GROUPS.keys()) + ['Any']

    def recommend_movies(
        self,
        user_text: str,
        user_genre: str,
        start_year: int,
        end_year: int,
        movie_region: str = "Any",
        top_n: int = 3
    ) -> pd.DataFrame:
        vec = self.vectorizer.transform([user_text])
        df = self.df.copy()

        # Genre filter
        if user_genre.lower() != 'any':
            mask_g = df['genres'].apply(
                lambda r: any(g['name'].lower() == user_genre.lower() for g in ast.literal_eval(r))
            )
        else:
            mask_g = pd.Series(True, index=df.index)

        # Region filter
        mr = movie_region.strip().lower()
        if mr != 'any':
            key = next((k for k in REGION_GROUPS if k.lower() == mr), None)
            if key:
                allowed = {c.lower() for c in REGION_GROUPS[key]}
                mask_r = df['production_countries'].apply(
                    lambda r: any(c['name'].lower() in allowed for c in ast.literal_eval(r))
                )
            else:
                mask_r = df['production_countries'].apply(
                    lambda r: any(c['name'].lower() == mr for c in ast.literal_eval(r))
                )
        else:
            mask_r = pd.Series(True, index=df.index)

        # Year filter
        df['release_year'] = pd.to_datetime(
            df['release_date'], errors='coerce'
        ).dt.year.fillna(0).astype(int)
        mask_y = df['release_year'].between(start_year, end_year)

        # Subset & drop zero revenue
        subset = df[mask_g & mask_r & mask_y].copy()
        subset = subset[subset['revenue'] > 0]
        if subset.empty:
            return pd.DataFrame(columns=[
                "original_title", "overview", "similarity",
                "release_year", "studio_size"
            ])

        # Similarity ranking
        sims = cosine_similarity(vec, self.X[subset.index]).flatten()
        subset['similarity'] = sims / (sims.max() or 1)

        # Pick top_n recommendations
        top = subset.sort_values('similarity', ascending=False).head(top_n).copy().reset_index(drop=True)

        # Guarantee one small-studio film
        if not (top['revenue'] < SMALL_THRESHOLD).any():
            smalls = subset[subset['revenue'] < SMALL_THRESHOLD]
            if not smalls.empty:
                top.iloc[-1] = smalls.sort_values('similarity', ascending=False).iloc[0]

        top['studio_size'] = top['revenue'].apply(
            lambda r: 'Small' if r < SMALL_THRESHOLD else 'Big'
        )
        return top[[
            "original_title", "overview",
            "similarity", "release_year", "studio_size"
        ]]
