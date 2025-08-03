import os
import pandas as pd
import ast
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from scipy import sparse

# === Constants & Region Mapping ===
SMALL_THRESHOLD = 2e7  # $20M threshold
REGION_GROUPS = {
    "North America": ["United States", "Canada", "Mexico"],
    "South America": ["Brazil", "Argentina", "Colombia", "Chile", "Peru", "Venezuela", "Ecuador", "Bolivia", "Paraguay", "Uruguay"],
    "Europe":        ["United Kingdom", "France", "Germany", "Italy", "Spain", "Netherlands", "Sweden", "Norway", "Denmark", "Finland", "Belgium", "Switzerland", "Austria", "Poland", "Russia"],
    "Asia":          ["Japan", "China", "India", "South Korea", "Hong Kong", "Taiwan", "Thailand", "Singapore", "Malaysia", "Indonesia"],
    "Africa":        ["Nigeria", "Egypt", "South Africa", "Kenya", "Morocco", "Algeria", "Ghana", "Ethiopia"],
    "Oceania":       ["Australia", "New Zealand", "Fiji"],
    "Latin America": ["Brazil", "Argentina", "Colombia", "Mexico", "Chile", "Peru", "Venezuela", "Ecuador"]
}

# 1. Load & feature-select
def load_data(path='tmdb_5000_movies.csv'):
    df = pd.read_csv(path)
    df.drop_duplicates(subset=['original_title'], keep='first', inplace=True)
    df['revenue'] = df.get('revenue', 0).fillna(0).astype(float)
    df = df[['id','original_title','overview','genres','keywords','production_countries','release_date','vote_average','revenue']]
    df.dropna(subset=['overview'], inplace=True)
    df['keywords'] = df['keywords'].fillna('[]')
    return df

# 2. Preprocessing helpers
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(tokens):
    tokens = [w.lower() for w in tokens if w.isalpha() or any(ch.isdigit() for ch in w)]
    return [stemmer.stem(w) for w in tokens if w not in stop_words]

def extract_keywords(text):
    try:
        parsed = ast.literal_eval(text)
        return ' '.join([kw['name'] for kw in parsed])
    except:
        return ''

def parse_list_of_dicts(text, key):
    try:
        parsed = ast.literal_eval(text)
        return [item.get(key, '') for item in parsed]
    except:
        return []

# 3. Build corpus & vectorizer
def build(df):
    df['tokenized_overview'] = df['overview'].apply(word_tokenize)
    df['clean_overview'] = df['tokenized_overview'].apply(preprocess).str.join(' ')
    df['keywords_text'] = df['keywords'].apply(extract_keywords)
    df['genre_list'] = df['genres'].apply(lambda x: parse_list_of_dicts(x, 'name'))
    df['country_list'] = df['production_countries'].apply(lambda x: parse_list_of_dicts(x, 'name'))
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    df['text_input'] = (
        df['original_title'] + ' ' +
        df['keywords_text'] + ' ' +
        df['clean_overview']
    )
    vectorizer = TfidfVectorizer(
        ngram_range=(1,3),
        min_df=3,
        max_df=0.8,
        sublinear_tf=True,
        max_features=30000,
        stop_words='english'
    )
    X = vectorizer.fit_transform(df['text_input'])
    return df, vectorizer, X

# 4. Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

def classify_tone(score):
    if score >= 0.3:
        return 'positive'
    elif score <= -0.3:
        return 'negative'
    else:
        return 'neutral'

# 5. Recommendation logic
def recommend_movies(
    user_text, user_genre, start_year, end_year, movie_region,
    df, vectorizer, X, top_n=10
):
    user_vec = vectorizer.transform([user_text])
    mask_g = df['genre_list'].apply(lambda L: user_genre in L) if user_genre != 'any' else True
    mask_y = df['release_year'].between(start_year, end_year)
    if movie_region in REGION_GROUPS:
        allowed = set(REGION_GROUPS[movie_region])
        mask_r = df['country_list'].apply(lambda L: any(c in allowed for c in L))
    else:
        mask_r = True

    subset = df[mask_g & mask_y & mask_r].copy()
    subset = subset[subset['revenue'] > 0]
    sims = cosine_similarity(user_vec, X[subset.index]).flatten()
    subset['similarity'] = sims / (sims.max() or 1)
    top = (
        subset
        .sort_values('similarity', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top['studio_size'] = top['revenue'].apply(lambda r: 'Small' if r < SMALL_THRESHOLD else 'Big')
    return top[['original_title','release_year','similarity','studio_size']]

# 6. Save artifacts
def save_artifacts(vectorizer, X, df, dir='artifacts'):
    os.makedirs(dir, exist_ok=True)
    joblib.dump(vectorizer, f"{dir}/tfidf_vectorizer.joblib")
    sparse.save_npz(f"{dir}/tfidf_matrix.npz", X)
    df.to_csv(f"{dir}/movies_metadata.csv", index=False)

if __name__ == '__main__':
    df = load_data()
    df, vectorizer, X = build(df)
    df['sentiment_score'] = df['text_input'].apply(get_sentiment)
    df['tone'] = df['sentiment_score'].apply(classify_tone)
    save_artifacts(vectorizer, X, df)
