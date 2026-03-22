import pandas as pd
import streamlit as st


def _top_counts(series: pd.Series, column_name: str, top_n: int = 10) -> pd.DataFrame:
    s = series.fillna("").astype(str).str.strip()
    s = s[s != ""]
    out = s.value_counts().head(top_n).reset_index()
    out.columns = [column_name, "count"]
    return out


def render_insights(df: pd.DataFrame) -> None:
    st.subheader("Dataset insights")

    if df is None or df.empty:
        st.info("No data available for insights.")
        return

    total_internships = len(df)
    unique_companies = df["company"].nunique()
    unique_profiles = df["profile"].nunique()
    unique_locations = df["location"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total internships", total_internships)
    c2.metric("Unique companies", unique_companies)
    c3.metric("Unique profiles", unique_profiles)
    c4.metric("Unique locations", unique_locations)

    left, right = st.columns(2)

    with left:
        st.markdown("### Top profiles")
        st.dataframe(
            _top_counts(df["profile"], "profile", top_n=10),
            use_container_width=True
        )

        st.markdown("### Top companies")
        st.dataframe(
            _top_counts(df["company"], "company", top_n=10),
            use_container_width=True
        )

    with right:
        st.markdown("### Top locations")
        st.dataframe(
            _top_counts(df["location"], "location", top_n=10),
            use_container_width=True
        )

        st.markdown("### Top skills")
        skills = (
            df["skills"]
            .fillna("")
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
        )
        skills = skills[skills != ""]
        top_skills = skills.value_counts().head(15).reset_index()
        top_skills.columns = ["skill", "count"]
        st.dataframe(top_skills, use_container_width=True)
