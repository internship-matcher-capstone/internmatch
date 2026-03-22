import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.preprocessing.preprocess import load_data, PROCESSED_CSV
from src.matching.recommender import get_recommendations
from src.insights.insights import render_insights


st.set_page_config(
    page_title="InternMatch",
    page_icon="🎯",
    layout="wide"
)


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_data(PROCESSED_CSV)


def fmt_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%d %b %Y")
    return str(value)


def show_result_card(row):
    with st.container(border=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            st.markdown(f"### {row.get('profile', '')}")
            st.write(f"**Company:** {row.get('company', '')}")
            st.write(f"**Location:** {row.get('location', '')}")
            st.write(f"**Start Date:** {fmt_value(row.get('start_date', ''))}")
            st.write(f"**Duration:** {row.get('duration', '')}")
            st.write(f"**Stipend:** {row.get('stipend', '')}")
            st.write(f"**Apply By:** {fmt_value(row.get('apply_by_date', ''))}")

        with col2:
            score = row.get("recommendation_score", 0.0)
            st.metric("Match", f"{score:.1f}%")

        skills = row.get("skills", "")
        education = row.get("education", "")
        perks = row.get("perks", "")
        offer = row.get("offer", "")

        if skills:
            st.write(f"**Skills:** {skills}")
        if education:
            st.write(f"**Education:** {education}")
        if perks:
            st.write(f"**Perks:** {perks}")
        if offer:
            st.write(f"**Offer:** {offer}")


def main():
    st.title("InternMatch")
    st.caption("Internship recommendation MVP for IIT Patna capstone")

    try:
        df = get_dataset()
    except Exception as e:
        st.error(f"Dataset load failed: {e}")
        st.stop()

    if df.empty:
        st.warning("The dataset loaded, but it is empty.")
        st.stop()

    st.success(f"Loaded {len(df)} internship records from {PROCESSED_CSV}")

    with st.sidebar:
        st.header("Search")

        query = st.text_area(
            "Enter skills / interests",
            placeholder="Python, SQL, machine learning, data analysis, cybersecurity",
            height=120
        )

        locations = ["All"] + sorted(
            x for x in df["location"].dropna().astype(str).str.strip().unique() if x
        )
        profiles = ["All"] + sorted(
            x for x in df["profile"].dropna().astype(str).str.strip().unique() if x
        )

        location_filter = st.selectbox("Location", locations, index=0)
        profile_filter = st.selectbox("Profile", profiles, index=0)
        top_n = st.slider("Number of results", min_value=5, max_value=25, value=10)

        run_search = st.button("Find internships", use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Recommendations", "Insights", "Dataset Preview"])

    with tab1:
        st.subheader("Recommended internships")

        if run_search:
            results = get_recommendations(
                df=df,
                query=query,
                top_n=top_n,
                location_filter=None if location_filter == "All" else location_filter,
                profile_filter=None if profile_filter == "All" else profile_filter,
            )

            if results.empty:
                st.info("No internships matched the current query/filters.")
            else:
                st.write(f"Showing top {len(results)} matches")
                for _, row in results.iterrows():
                    show_result_card(row)
        else:
            st.info("Enter your skills in the sidebar and click **Find internships**.")

    with tab2:
        render_insights(df)

    with tab3:
        st.subheader("Dataset preview")
        st.dataframe(df.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
