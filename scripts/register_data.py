import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
DATASET_PATH = DATA_DIR / "tourism.csv"

REQUIRED_COLUMNS = [
    "Age",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "ProdTaken",
]


def generate_sample_data() -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(42)
    n = 500

    data = {
        "Age": rng.integers(18, 70, size=n),
        "CityTier": rng.integers(1, 4, size=n),
        "Occupation": rng.choice(["Salaried", "Small Business", "Large Business", "Free Lancer"], size=n),
        "Gender": rng.choice(["Male", "Female"], size=n),
        "NumberOfPersonVisiting": rng.integers(1, 7, size=n),
        "PreferredPropertyStar": rng.choice([3, 4, 5], size=n),
        "MaritalStatus": rng.choice(["Single", "Married", "Divorced", "Unmarried"], size=n),
        "NumberOfTrips": rng.integers(0, 10, size=n),
        "Passport": rng.integers(0, 2, size=n),
        "OwnCar": rng.integers(0, 2, size=n),
        "NumberOfChildrenVisiting": rng.integers(0, 5, size=n),
        "Designation": rng.choice(["Executive", "Manager", "Senior Manager", "AVP", "VP"], size=n),
        "MonthlyIncome": rng.integers(10000, 150000, size=n),
    }
    data["ProdTaken"] = (
        (data["Age"] > 25)
        & (data["NumberOfTrips"] > 1)
        & (data["Passport"] == 1)
        & (data["MonthlyIncome"] > 20000)
    ).astype(int)
    return pd.DataFrame(data)


def validate_dataset(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    empty_count = df.isna().sum().sum()
    if empty_count:
        raise ValueError(f"Dataset contains {empty_count} null/empty values.")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        sample_df = generate_sample_data()
        sample_df.to_csv(DATASET_PATH, index=False)
        print(f"Generated synthetic dataset at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    validate_dataset(df)

    report = {
        "dataset_path": str(DATASET_PATH),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "required_columns_present": all(column in df.columns for column in REQUIRED_COLUMNS),
        "status": "passed",
    }

    with (REPORTS_DIR / "data_validation_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Validated dataset: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
