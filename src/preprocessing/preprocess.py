from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

RAW_CSV = DATA_DIR / "reference_from_replit" / "merged_internships_dataset.csv"
PROCESSED_CSV = DATA_DIR / "processed" / "internships_clean.csv"

REQUIRED_COLUMNS = [
    "internship_id",
    "date_time",
    "profile",
    "company",
    "location",
    "start_date",
    "stipend",
    "duration",
    "apply_by_date",
    "offer",
    "education",
    "skills",
    "perks",
]

TEXT_COLUMNS = [
    "internship_id",
    "profile",
    "company",
    "location",
    "start_date",
    "stipend",
    "duration",
    "offer",
    "education",
    "skills",
    "perks",
]


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    for col in TEXT_COLUMNS:
        df[col] = _clean_text(df[col])

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["apply_by_date"] = pd.to_datetime(df["apply_by_date"], errors="coerce")

    return df


def load_data(csv_path=PROCESSED_CSV) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.is_absolute():
        path = ROOT_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_dataframe(df)
    return df


def load_and_preprocess(csv_path=PROCESSED_CSV) -> pd.DataFrame:
    return load_data(csv_path)
