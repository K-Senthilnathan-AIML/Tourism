import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "tourism.csv"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "data"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"

FEATURE_COLUMNS = [
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
]
TARGET_COLUMN = "ProdTaken"


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run scripts/register_data.py first.")

    df = pd.read_csv(DATASET_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_df.to_csv(ARTIFACTS_DIR / "train.csv", index=False)
    test_df.to_csv(ARTIFACTS_DIR / "test.csv", index=False)

    with FEATURES_PATH.open("w", encoding="utf-8") as file:
        json.dump(FEATURE_COLUMNS, file, indent=2)

    print(f"Prepared train/test datasets in {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
