import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from preprocess import load_data, load_and_preprocess, RAW_CSV, PROCESSED_CSV
from recommender import get_recommendations, tokenize_query
from insights import render_insights

st.set_page_config(
    page_title="Internship Matcher (India 2025)",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data(show_spinner="Loading internship data...")
def get_data():
    df = load_data()
    return df


def show_no_data_warning():
    st.error("Dataset not found!")
    st.markdown("""
    Please upload your dataset file to get started:

    **Expected file path:** `data/merged_internships_dataset.csv`

    Place the CSV file in the `data/` directory relative to this app, then refresh the page.
    """)
    st.stop()


def matcher_page(df):
    st.title("🎓 Internship Matcher (India 2025)")
    st.markdown("Find internships that match your skills and preferences with explainable recommendations.")

    with st.sidebar:
        st.header("Your Preferences")

        skills_input = st.text_input(
            "Skills (required)",
            placeholder="python, sql, excel, machine learning",
            help="Enter skills separated by commas"
        )

        location_mode = st.radio(
            "Location Preference",
            options=["Any", "Remote/WFH only", "City"],
            index=0
        )

        city_filter = ""
        if location_mode == "City":
            common_cities = sorted([
                "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
                "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Noida",
                "Gurgaon", "Chandigarh", "Indore", "Bhopal", "Lucknow"
            ])
            city_choice = st.selectbox("Select City", options=["-- Select --"] + common_cities + ["Other (type below)"])
            if city_choice == "Other (type below)" or city_choice == "-- Select --":
                city_filter = st.text_input("Enter city name", placeholder="e.g., Nagpur")
            else:
                city_filter = city_choice

        min_stipend = st.number_input(
            "Minimum Stipend (INR/month)",
            min_value=0,
            max_value=200000,
            value=0,
            step=1000,
            help="Enter 0 to include all stipend levels"
        )

        duration_range = st.slider(
            "Duration (months)",
            min_value=1,
            max_value=24,
            value=(1, 12),
            step=1
        )
        duration_min, duration_max = duration_range

        include_unpaid = st.checkbox("Include unpaid internships", value=False)

        sort_mode = st.selectbox(
            "Sort results by",
            options=["Best match", "Highest stipend", "Closest deadline"]
        )

        search_btn = st.button("Find Internships", type="primary", use_container_width=True)

    if not search_btn:
        st.info("Use the sidebar to enter your skills and preferences, then click **Find Internships**.")
        st.markdown("""
        **How it works:**
        1. Enter your skills (e.g., `python, data analysis, excel`)
        2. Set your location, stipend and duration filters
        3. Click **Find Internships** to get your top 10 matches with explanations
        """)
        return

    if not skills_input.strip():
        st.warning("Please enter at least one skill to get recommendations.")
        return

    with st.spinner("Finding your best matches..."):
        results, explanations = get_recommendations(
            df=df,
            skills_query=skills_input,
            location_mode=location_mode,
            city_filter=city_filter,
            min_stipend=min_stipend,
            duration_min=duration_min,
            duration_max=duration_max,
            include_unpaid=include_unpaid,
            sort_mode=sort_mode
        )

    if results.empty:
        st.error("No internships found matching your filters.")
        st.markdown("""
        **Suggestions to broaden your search:**
        - Try setting **Location** to "Any"
        - Lower the **Minimum Stipend** to 0
        - Widen the **Duration** range
        - Check **Include unpaid internships**
        - Simplify your skills list
        """)
        return

    st.success(f"Found **{len(results)}** matching internships for skills: `{skills_input}`")

    user_tokens = tokenize_query(skills_input)

    for i, (idx, row) in enumerate(results.iterrows()):
        exp = explanations[i]
        with st.expander(
            f"#{i+1} — {row.get('profile', 'N/A')} at {row.get('company', 'N/A')}",
            expanded=(i < 3)
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Company:** {row.get('company', 'N/A')}")
                st.markdown(f"**Role:** {row.get('profile', 'N/A')}")
                loc = row.get('location', 'N/A')
                st.markdown(f"**Location:** {loc}")
            with col2:
                stipend_orig = row.get('stipend', 'N/A')
                stipend_min_v = row.get('stipend_min')
                stipend_max_v = row.get('stipend_max')
                stipend_type = row.get('stipend_type', 'unknown')
                if pd.notna(stipend_min_v) and pd.notna(stipend_max_v):
                    if stipend_type == "fixed":
                        parsed_stip = f"₹{int(stipend_max_v):,}"
                    elif stipend_type == "unpaid":
                        parsed_stip = "Unpaid"
                    else:
                        parsed_stip = f"₹{int(stipend_min_v):,} – ₹{int(stipend_max_v):,}"
                else:
                    parsed_stip = stipend_type.replace("_", " ").title()
                st.markdown(f"**Stipend:** {stipend_orig}")
                st.markdown(f"**Parsed Stipend:** {parsed_stip}")
                st.markdown(f"**Duration:** {row.get('duration', 'N/A')}")
                dur_m = row.get('duration_months')
                if pd.notna(dur_m):
                    st.markdown(f"**Duration (months):** {float(dur_m):.1f}")
            with col3:
                st.markdown(f"**Apply By:** {row.get('apply_by_date', 'N/A')}")
                st.markdown(f"**Offer:** {row.get('offer', 'N/A')}")
                st.markdown(f"**Education:** {row.get('education', 'N/A')}")

            st.markdown("---")
            why_parts = []
            if exp["matched_skills"]:
                why_parts.append(f"Matched skills: **{', '.join(exp['matched_skills'])}**")
            if exp["constraints"]:
                why_parts.append(f"Satisfies: {', '.join(exp['constraints'])}")
            if why_parts:
                st.markdown("**Why this matched:** " + " | ".join(why_parts))
            else:
                st.markdown("**Why this matched:** General profile and text similarity to your query.")

            score_col1, score_col2 = st.columns(2)
            with score_col1:
                st.metric("Relevance Score", f"{exp['similarity']:.2%}")
            with score_col2:
                st.metric("Final Score", f"{exp['final_score']:.2%}")

    st.divider()
    display_cols = ["profile", "company", "location", "stipend", "stipend_min",
                    "stipend_max", "duration", "duration_months", "apply_by_date", "offer"]
    available_cols = [c for c in display_cols if c in results.columns]
    download_df = results[available_cols].copy()
    csv = download_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="internship_matches.csv",
        mime="text/csv"
    )


def about_page():
    st.title("About — Internship Matcher (India 2025)")
    st.markdown("""
    ## What is this app?

    The **Internship Matcher (India 2025)** is an AI-powered recommendation tool that helps students and graduates discover 
    the most relevant internship opportunities across India — with full transparency about *why* each internship was recommended.

    ---

    ## How it works

    ### Data
    The app loads a dataset of ~8,400 internship listings scraped from Indian internship platforms. 
    On first launch, it preprocesses the raw data and caches it for fast subsequent loads.

    ### Preprocessing
    The app cleans and structures messy text fields:
    - **Stipend parsing** — Extracts minimum/maximum values and classifies as fixed, range, unpaid, or performance-based
    - **Duration parsing** — Converts "3 months", "8 weeks", "60 days" into a uniform numeric value
    - **Location parsing** — Detects remote/WFH listings, extracts cities, and flags multi-city roles
    - **Date parsing** — Safely parses application deadlines, posting dates, and start dates
    - **Skills tokenization** — Splits skills into individual tokens for matching

    ### Recommendation Algorithm
    Recommendations use **TF-IDF** (Term Frequency–Inverse Document Frequency) and **cosine similarity** 
    to find internships whose full description best matches your skill set.

    The final score combines:
    - **75% relevance** — How well the internship text matches your skills
    - **15% stipend score** — Higher stipends rank better within your filter
    - **10% deadline score** — Closer upcoming deadlines get a slight boost

    ### Explainability
    For every recommendation, the app shows:
    - Which of your skills were explicitly matched
    - What constraints (location, stipend, duration) the internship satisfies
    - The individual relevance and final scores

    ---

    ## Pages

    | Page | Purpose |
    |------|---------|
    | **Matcher** | Enter your profile and get top 10 recommendations |
    | **Market Insights** | Charts and statistics about the internship landscape |
    | **About** | This page |

    ---

    ## Tech Stack
    - **Python** + **Streamlit** for the UI
    - **pandas** + **numpy** for data processing
    - **scikit-learn** for TF-IDF and cosine similarity
    - **python-dateutil** for robust date parsing
    """)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Matcher", "Market Insights", "About"],
        index=0
    )

    df = get_data()

    if df is None:
        if page == "About":
            about_page()
            return
        show_no_data_warning()

    if page == "Matcher":
        matcher_page(df)
    elif page == "Market Insights":
        render_insights(df)
    elif page == "About":
        about_page()


if __name__ == "__main__":
    main()
