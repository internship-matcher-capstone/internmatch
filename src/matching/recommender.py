import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

TODAY = pd.Timestamp.now()
DEADLINE_WINDOW_DAYS = 60


@st.cache_resource
def build_tfidf_model(texts):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def tokenize_query(query):
    tokens = re.split(r"[,;|]", query)
    return [t.strip().lower() for t in tokens if t.strip()]


def compute_deadline_score(dt_series):
    scores = pd.Series(0.5, index=dt_series.index)
    valid = dt_series.notna()
    future = dt_series[valid] > TODAY
    past = dt_series[valid] <= TODAY

    future_dates = dt_series[valid & (dt_series > TODAY)]
    if not future_dates.empty:
        days_left = (future_dates - TODAY).dt.days.clip(0, DEADLINE_WINDOW_DAYS)
        scores[future_dates.index] = 1.0 - (days_left / DEADLINE_WINDOW_DAYS) * 0.5
    scores[valid & past] = 0.0
    return scores


def compute_stipend_score(df_filtered):
    scores = pd.Series(0.5, index=df_filtered.index)
    known = df_filtered["stipend_max"].notna()
    if known.sum() > 0:
        vals = df_filtered.loc[known, "stipend_max"]
        max_val = vals.max()
        if max_val > 0:
            scores[known] = vals / max_val
        else:
            scores[known] = 0.0
    return scores


def filter_internships(df, skills_query, location_mode, city_filter, min_stipend,
                        duration_min, duration_max, include_unpaid):
    mask = pd.Series(True, index=df.index)

    if location_mode == "Remote/WFH only":
        mask &= df["is_remote"] == True
    elif location_mode == "City" and city_filter:
        city_lower = city_filter.lower()
        city_mask = (
            df["city"].fillna("").str.lower().str.contains(city_lower, na=False)
            | df["is_remote"] == True
        )
        mask &= city_mask

    stipend_mask = pd.Series(True, index=df.index)
    if not include_unpaid:
        unpaid_mask = df["stipend_type"] == "unpaid"
        stipend_mask &= ~unpaid_mask

    if min_stipend > 0:
        known_stip = df["stipend_max"].notna()
        stip_ok = df["stipend_max"].fillna(0) >= min_stipend
        unknown_stip = df["stipend_max"].isna() & (df["stipend_type"].isin(["unknown", None]) | df["stipend_type"].isna())
        stipend_mask &= (stip_ok | unknown_stip) | (include_unpaid & (df["stipend_type"] == "unpaid"))

    mask &= stipend_mask

    dur_known = df["duration_months"].notna()
    dur_ok = (df["duration_months"] >= duration_min) & (df["duration_months"] <= duration_max)
    dur_unknown = df["duration_months"].isna()
    mask &= dur_ok | dur_unknown

    return df[mask].copy()


def rank_internships(df_filtered, skills_query, vectorizer, tfidf_matrix, sort_mode, original_df):
    if df_filtered.empty:
        return df_filtered

    filtered_idx = df_filtered.index
    sub_matrix = tfidf_matrix[filtered_idx]

    if skills_query.strip():
        query_vec = vectorizer.transform([skills_query.lower()])
        similarities = cosine_similarity(query_vec, sub_matrix).flatten()
    else:
        similarities = np.zeros(len(df_filtered))

    sim_series = pd.Series(similarities, index=filtered_idx)

    stipend_scores = compute_stipend_score(df_filtered)
    deadline_scores = compute_deadline_score(df_filtered["apply_by_dt"])

    final_scores = 0.75 * sim_series + 0.15 * stipend_scores + 0.10 * deadline_scores

    df_filtered = df_filtered.copy()
    df_filtered["_similarity"] = sim_series.values
    df_filtered["_final_score"] = final_scores.values

    if sort_mode == "Best match":
        df_filtered = df_filtered.sort_values("_final_score", ascending=False)
    elif sort_mode == "Highest stipend":
        df_filtered = df_filtered.sort_values("stipend_max", ascending=False, na_position="last")
    elif sort_mode == "Closest deadline":
        df_filtered["_days_left"] = (df_filtered["apply_by_dt"] - TODAY).dt.days
        df_filtered = df_filtered.sort_values("_days_left", ascending=True, na_position="last")

    return df_filtered.head(10)


def explain_match(row, user_skill_tokens):
    matched = []
    row_tokens = row.get("skills_tokens", [])
    if isinstance(row_tokens, str):
        row_tokens = eval(row_tokens) if row_tokens.startswith("[") else []

    for tok in user_skill_tokens:
        if any(tok in rt or rt in tok for rt in row_tokens):
            matched.append(tok)

    constraints = []
    if row.get("is_remote"):
        constraints.append("Remote/WFH")
    if pd.notna(row.get("stipend_max")):
        constraints.append(f"Stipend: ₹{int(row['stipend_max']):,}")
    if pd.notna(row.get("duration_months")):
        constraints.append(f"Duration: {row['duration_months']:.1f} months")

    return matched, constraints


def get_recommendations(df, skills_query, location_mode, city_filter, min_stipend,
                         duration_min, duration_max, include_unpaid, sort_mode):
    vectorizer, tfidf_matrix = build_tfidf_model(tuple(df["text_blob"].fillna("").tolist()))

    df_filtered = filter_internships(
        df, skills_query, location_mode, city_filter,
        min_stipend, duration_min, duration_max, include_unpaid
    )

    if df_filtered.empty:
        return df_filtered, []

    results = rank_internships(df_filtered, skills_query, vectorizer, tfidf_matrix, sort_mode, df)

    user_tokens = tokenize_query(skills_query)
    explanations = []
    for _, row in results.iterrows():
        matched_skills, constraints = explain_match(row, user_tokens)
        explanations.append({
            "matched_skills": matched_skills,
            "constraints": constraints,
            "similarity": row.get("_similarity", 0.0),
            "final_score": row.get("_final_score", 0.0),
        })

    return results, explanations
