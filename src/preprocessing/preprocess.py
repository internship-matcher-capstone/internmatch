import os
import re
import pandas as pd
import numpy as np
from dateutil import parser as date_parser

RAW_CSV = os.path.join(os.path.dirname(__file__), "data", "merged_internships_dataset.csv")
PROCESSED_CSV = os.path.join(os.path.dirname(__file__), "data", "internships_processed.csv")

MISSING_LIKE = {"n/a", "na", "not specified", "not available", "none", "null", "-", "", "nan", "not mentioned", "n.a.", "not applicable"}


def normalize_missing(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.lower() in MISSING_LIKE:
        return None
    return s if s else None


def parse_stipend(raw):
    result = {
        "stipend_min": None,
        "stipend_max": None,
        "stipend_type": "unknown",
        "stipend_period": "unknown",
    }
    if raw is None:
        return result
    text = raw.lower()

    if "unpaid" in text:
        result["stipend_min"] = 0.0
        result["stipend_max"] = 0.0
        result["stipend_type"] = "unpaid"
        return result

    if "performance" in text:
        result["stipend_type"] = "performance_based"

    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    nums = []
    for n in numbers:
        try:
            nums.append(float(n.replace(",", "")))
        except ValueError:
            pass

    if nums:
        result["stipend_min"] = min(nums)
        result["stipend_max"] = max(nums)
        if result["stipend_type"] != "performance_based":
            if len(nums) == 1:
                result["stipend_type"] = "fixed"
            else:
                result["stipend_type"] = "range"

    if "month" in text or "/month" in text or "pm" in text:
        result["stipend_period"] = "month"
    elif "week" in text:
        result["stipend_period"] = "week"
    elif "lump" in text or "one time" in text or "one-time" in text:
        result["stipend_period"] = "lump_sum"

    return result


def parse_duration(raw):
    if raw is None:
        return None
    text = raw.lower()
    months_m = re.search(r"(\d+(?:\.\d+)?)\s*month", text)
    weeks_m = re.search(r"(\d+(?:\.\d+)?)\s*week", text)
    days_m = re.search(r"(\d+(?:\.\d+)?)\s*day", text)
    if months_m:
        return float(months_m.group(1))
    if weeks_m:
        return float(weeks_m.group(1)) / 4.345
    if days_m:
        return float(days_m.group(1)) / 30.0
    return None


def safe_parse_date(val):
    if val is None:
        return pd.NaT
    try:
        return pd.to_datetime(date_parser.parse(str(val), fuzzy=True))
    except Exception:
        return pd.NaT


def parse_location(raw):
    result = {
        "is_remote": False,
        "city": "Unknown",
        "is_multi_city": False,
    }
    if raw is None:
        return result
    text = raw.lower()

    if any(k in text for k in ["work from home", "wfh", "remote", "work from office and home"]):
        result["is_remote"] = True

    if "multiple" in text or "pan india" in text or "all india" in text:
        result["is_multi_city"] = True
        result["city"] = "Multiple"
        return result

    separators = re.split(r"[,/|&]", raw)
    clean_parts = [p.strip() for p in separators if p.strip()]
    if len(clean_parts) > 2:
        result["is_multi_city"] = True
        result["city"] = "Multiple"
    elif len(clean_parts) == 1:
        city_text = clean_parts[0].strip()
        if city_text.lower() not in ["work from home", "wfh", "remote", "online", ""]:
            result["city"] = city_text.title()
        elif result["is_remote"]:
            result["city"] = "Remote"
    elif len(clean_parts) == 2:
        city_text = clean_parts[0].strip()
        if city_text.lower() not in ["work from home", "wfh", "remote", "online", ""]:
            result["city"] = city_text.title()
            result["is_multi_city"] = True
        else:
            result["city"] = "Remote"
    else:
        if result["is_remote"]:
            result["city"] = "Remote"

    return result


def tokenize_skills(raw):
    if raw is None:
        return []
    tokens = re.split(r"[,|;]", raw)
    return [t.strip().lower() for t in tokens if t.strip()]


def build_text_blob(row):
    parts = []
    for col in ["profile", "skills", "perks", "education", "offer", "company"]:
        val = row.get(col)
        if val and not pd.isna(val):
            parts.append(str(val))
    return " ".join(parts).lower()


def load_and_preprocess(raw_path=RAW_CSV, processed_path=PROCESSED_CSV):
    df = pd.read_csv(raw_path, dtype=str, low_memory=False)

    text_cols = ["profile", "company", "location", "start_date", "stipend",
                 "duration", "apply_by_date", "offer", "education", "skills", "perks", "date_time"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_missing)

    stipend_parsed = df["stipend"].apply(parse_stipend).apply(pd.Series)
    df = pd.concat([df, stipend_parsed], axis=1)

    df["duration_months"] = df["duration"].apply(parse_duration)

    df["apply_by_dt"] = df["apply_by_date"].apply(safe_parse_date)
    df["posted_dt"] = df["date_time"].apply(safe_parse_date)

    df["start_immediate"] = df["start_date"].apply(
        lambda x: bool(x and "immediate" in x.lower())
    )
    df["start_dt"] = df["start_date"].apply(
        lambda x: safe_parse_date(x) if x and "immediate" not in x.lower() else pd.NaT
    )

    loc_parsed = df["location"].apply(parse_location).apply(pd.Series)
    df = pd.concat([df, loc_parsed], axis=1)

    df["skills_tokens"] = df["skills"].apply(tokenize_skills)
    df["text_blob"] = df.apply(build_text_blob, axis=1)

    df.to_csv(processed_path, index=False)
    return df


def load_data():
    if os.path.exists(PROCESSED_CSV):
        df = pd.read_csv(PROCESSED_CSV, dtype=str, low_memory=False)
        float_cols = ["stipend_min", "stipend_max", "duration_months"]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        bool_cols = ["is_remote", "start_immediate", "is_multi_city"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].map({"True": True, "False": False, True: True, False: False}).fillna(False)
        dt_cols = ["apply_by_dt", "posted_dt", "start_dt"]
        for col in dt_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df["skills_tokens"] = df["skills_tokens"].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else []
        )
        if "text_blob" not in df.columns:
            df["text_blob"] = df.apply(build_text_blob, axis=1)
        return df
    else:
        if not os.path.exists(RAW_CSV):
            return None
        return load_and_preprocess()
