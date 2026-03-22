import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize_query(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_+.#-]+", str(text).lower())


def _build_search_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["profile"].fillna("") + " "
        + df["skills"].fillna("") + " "
        + df["skills"].fillna("") + " "
        + df["education"].fillna("") + " "
        + df["company"].fillna("") + " "
        + df["location"].fillna("") + " "
        + df["perks"].fillna("") + " "
        + df["offer"].fillna("")
    ).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def get_recommendations(
    df: pd.DataFrame,
    query: str = "",
    top_n: int = 10,
    location_filter: str | None = None,
    profile_filter: str | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    work_df = df.copy()

    if location_filter:
        work_df = work_df[
            work_df["location"].astype(str).str.lower() == location_filter.lower()
        ]

    if profile_filter:
        work_df = work_df[
            work_df["profile"].astype(str).str.lower() == profile_filter.lower()
        ]

    if work_df.empty:
        return work_df

    work_df["search_text"] = _build_search_text(work_df)

    if not str(query).strip():
        work_df["recommendation_score"] = 0.0
        if "date_time" in work_df.columns:
            work_df = work_df.sort_values("date_time", ascending=False)
        return work_df.head(top_n).drop(columns=["search_text"])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    doc_matrix = vectorizer.fit_transform(work_df["search_text"].tolist())
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(doc_matrix, query_vector).ravel()
    work_df["recommendation_score"] = (scores * 100).round(2)

    sort_cols = ["recommendation_score"]
    ascending = [False]

    if "date_time" in work_df.columns:
        sort_cols.append("date_time")
        ascending.append(False)

    work_df = work_df.sort_values(sort_cols, ascending=ascending)

    return work_df.head(top_n).drop(columns=["search_text"])
