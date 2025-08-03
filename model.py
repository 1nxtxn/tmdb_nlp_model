import os
import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from scipy import sparse

# === Config ===
SMALL_THRESHOLD = 2e7
REGION_GROUPS = {
    "North America": ["United States","Canada","Mexico"],
    "South America": ["Brazil","Argentina","Colombia","Chile","Peru","Venezuela","Ecuador","Bolivia","Paraguay","Uruguay"],
    "Europe":        ["United Kingdom","France","Germany","Italy","Spain","Netherlands","Sweden","Norway","Denmark","Finland","Belgium","Switzerland","Austria","Poland","Russia"],
    "Asia":          ["Japan","China","India","South Korea","Hong Kong","Taiwan","Thailand","Singapore","Malaysia","Indonesia"],
    "Africa":        ["Nigeria","Egypt","South Africa","Kenya","Morocco","Algeria","Ghana","Ethiopia"],
    "Oceania":       ["Australia","New Zealand","Fiji"],
    "Latin America": ["Brazil","Argentina","Colombia","Mexico","Chile","Peru","Venezuela","Ecuador"]
}

def load_data(path="tmdb_5000_movies.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop_duplicates(subset=["original_title"], keep="first", inplace=True)
    df["revenue"] = df.get("revenue", 0).fillna(0).astype(float)
    df = df[[
        "id","original_title","overview","genres","keywords",
        "production_countries","release_date","vote_average","revenue"
    ]]
    df.dropna(subset=["overview"], inplace=True)
    df["keywords"] = df["keywords"].fillna("[]")
    return df

def build(df: pd.DataFrame):
    # parse JSON-like columns
    df["genre_list"]    = df["genres"].apply(lambda x: [i["name"] for i in ast.literal_eval(x)])
    df["country_list"]  = df["production_countries"].apply(lambda x: [i["name"] for i in ast.literal_eval(x)])
    df["release_year"]  = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)

    # create the text for TF-IDF
    df["text_input"] = (
        df["original_title"] + " "
        + df["overview"]      + " "
        + df["keywords"].apply(lambda t: " ".join([i["name"] for i in ast.literal_eval(t)]))
    )

    # build TF-IDF
    tfidf = TfidfVectorizer(
        ngram_range=(1,3),
        min_df=3,
        max_df=0.8,
        sublinear_tf=True,
        max_features=30000,
        stop_words="english"
    )
    X = tfidf.fit_transform(df["text_input"])
    return df, tfidf, X

def recommend_movies(
    user_text: str,
    user_genre: str,
    start_year: int,
    end_year: int,
    movie_region: str,
    df: pd.DataFrame,
    tfidf: TfidfVectorizer,
    X,
    top_n: int = 10
) -> pd.DataFrame:
    vec = tfidf.transform([user_text])

    # filters
    mask_g = df["genre_list"].apply(lambda L: user_genre in L) if user_genre != "any" else True
    mask_y = df["release_year"].between(start_year, end_year)
    if movie_region in REGION_GROUPS:
        allowed = set(REGION_GROUPS[movie_region])
        mask_r = df["country_list"].apply(lambda L: any(c in allowed for c in L))
    else:
        mask_r = True

    sub = df[mask_g & mask_y & mask_r].copy()
    sub = sub[sub["revenue"] > 0]

    # similarity
    sims = cosine_similarity(vec, X[sub.index]).flatten()
    sub["similarity"] = sims / (sims.max() or 1)

    # pick top_n
    top = (
        sub
        .sort_values("similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top["studio_size"] = top["revenue"].apply(lambda r: "Small" if r < SMALL_THRESHOLD else "Big")
    return top[["original_title","release_year","similarity","studio_size"]]

def save_artifacts(tfidf, X, df, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(tfidf,  f"{out_dir}/tfidf_vectorizer.joblib")
    sparse.save_npz(f"{out_dir}/tfidf_matrix.npz",     X)
    df.to_csv(f"{out_dir}/movies_metadata.csv", index=False)

if __name__ == "__main__":
    df, tfidf_model, X = build(load_data())
    save_artifacts(tfidf_model, X, df)
