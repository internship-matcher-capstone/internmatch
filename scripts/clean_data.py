from pathlib import Path
import pandas as pd
import re
import csv

INPUT_FILE = Path("data/raw/merged/merged_internships_dataset.csv")
OUTPUT_FILE = Path("data/processed/internships_clean.csv")


def to_snake_case(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text


def clean_text(x):
    if pd.isna(x):
        return x
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Original shape:", df.shape)
    print("Original columns:", list(df.columns))

    # Clean column names
    df.columns = [to_snake_case(col) for col in df.columns]

    # Clean text cells
    for col in df.columns:
        df[col] = df[col].apply(clean_text)

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Parse datetime if present
    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")

    print("Cleaned shape:", df.shape)
    print("Saved to:", OUTPUT_FILE)
    print(df.head())


if __name__ == "__main__":
    main()