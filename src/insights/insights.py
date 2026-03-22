import pandas as pd
import numpy as np
import streamlit as st
import ast


def safe_tokens(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.startswith("["):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return []


def render_insights(df):
    st.header("Market Insights")
    st.subheader("Key Observations")

    insights = generate_insights(df)
    for insight in insights:
        st.markdown(f"- {insight}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Cities by Internship Count")
        city_counts = df["city"].value_counts().head(12).reset_index()
        city_counts.columns = ["City", "Count"]
        city_counts = city_counts[city_counts["City"] != "Unknown"]
        st.bar_chart(city_counts.set_index("City"))

    with col2:
        st.subheader("Remote vs On-site")
        remote_counts = df["is_remote"].value_counts().reset_index()
        remote_counts.columns = ["Type", "Count"]
        remote_counts["Type"] = remote_counts["Type"].map({True: "Remote/WFH", False: "On-site"})
        st.bar_chart(remote_counts.set_index("Type"))

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top Internship Profiles")
        profiles = df["profile"].value_counts().head(10).reset_index()
        profiles.columns = ["Profile", "Count"]
        st.bar_chart(profiles.set_index("Profile"))

    with col4:
        st.subheader("Stipend Type Distribution")
        stip_types = df["stipend_type"].fillna("unknown").value_counts().reset_index()
        stip_types.columns = ["Stipend Type", "Count"]
        st.bar_chart(stip_types.set_index("Stipend Type"))

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Top Skills in Demand")
        all_tokens = []
        for tokens in df["skills_tokens"].apply(safe_tokens):
            all_tokens.extend(tokens)
        if all_tokens:
            skills_series = pd.Series(all_tokens).value_counts().head(15).reset_index()
            skills_series.columns = ["Skill", "Count"]
            st.bar_chart(skills_series.set_index("Skill"))
        else:
            st.info("No skills data available.")

    with col6:
        st.subheader("Duration Distribution (Months)")
        dur_data = df["duration_months"].dropna()
        if not dur_data.empty:
            bins = [0, 1, 2, 3, 4, 5, 6, 9, 12, 24]
            labels = ["<1", "1", "2", "3", "4", "5", "6", "7-9", "10-12", "12+"]
            binned = pd.cut(dur_data, bins=bins, labels=labels[1:], right=True)
            bin_counts = binned.value_counts().sort_index().reset_index()
            bin_counts.columns = ["Duration", "Count"]
            st.bar_chart(bin_counts.set_index("Duration"))
        else:
            st.info("No duration data available.")

    st.subheader("City-wise Average Stipend (Top Cities)")
    city_stip = (
        df[df["stipend_max"].notna() & (df["city"] != "Unknown") & (df["city"] != "Multiple")]
        .groupby("city")["stipend_max"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    city_stip.columns = ["City", "Avg Stipend (INR)"]
    city_stip["Avg Stipend (INR)"] = city_stip["Avg Stipend (INR)"].round(0)
    if not city_stip.empty:
        st.bar_chart(city_stip.set_index("City"))
    else:
        st.info("Not enough stipend data for city-wise comparison.")


def generate_insights(df):
    insights = []
    total = len(df)
    insights.append(f"The dataset contains **{total:,}** internship listings across India.")

    remote_pct = (df["is_remote"].sum() / total * 100) if total > 0 else 0
    insights.append(f"**{remote_pct:.1f}%** of internships are remote/WFH, showing strong remote culture post-pandemic.")

    top_city = df[df["city"] != "Unknown"]["city"].value_counts().idxmax() if (df["city"] != "Unknown").any() else "N/A"
    insights.append(f"**{top_city}** leads as the city with the most internship opportunities.")

    top_profile = df["profile"].value_counts().idxmax() if df["profile"].notna().any() else "N/A"
    insights.append(f"**{top_profile}** is the most common internship profile, reflecting high demand in that domain.")

    all_tokens = []
    for tokens in df["skills_tokens"].apply(safe_tokens):
        all_tokens.extend(tokens)
    if all_tokens:
        top_skill = pd.Series(all_tokens).value_counts().idxmax()
        insights.append(f"**{top_skill.title()}** is the most in-demand skill across all internships.")

    paid_pct = (df["stipend_type"].isin(["fixed", "range"]).sum() / total * 100) if total > 0 else 0
    unpaid_pct = (df["stipend_type"] == "unpaid").sum() / total * 100 if total > 0 else 0
    insights.append(f"**{paid_pct:.1f}%** of internships offer a fixed or range-based stipend, while **{unpaid_pct:.1f}%** are unpaid.")

    known_stip = df["stipend_max"].dropna()
    if not known_stip.empty:
        median_stip = known_stip.median()
        max_stip = known_stip.max()
        insights.append(f"Median stipend (where disclosed) is **₹{int(median_stip):,}/month**, with the highest at **₹{int(max_stip):,}**.")

    dur_known = df["duration_months"].dropna()
    if not dur_known.empty:
        avg_dur = dur_known.mean()
        insights.append(f"Average internship duration is **{avg_dur:.1f} months**, helping plan academic calendars.")

    apply_known = df["apply_by_dt"].dropna()
    if not apply_known.empty:
        future_deadlines = apply_known[apply_known > pd.Timestamp.now()]
        insights.append(f"**{len(future_deadlines):,}** internships still have open application deadlines — apply quickly!")

    multi_city_pct = df["is_multi_city"].sum() / total * 100 if total > 0 else 0
    insights.append(f"**{multi_city_pct:.1f}%** of internships are available across multiple cities or locations, offering geographical flexibility.")

    top5_profiles = df["profile"].value_counts().head(5).index.tolist()
    insights.append(f"Top 5 most common profiles: {', '.join(f'**{p}**' for p in top5_profiles)}.")

    return insights
