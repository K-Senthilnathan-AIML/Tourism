import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "artifacts" / "data" / "train.csv"
TEST_PATH = REPO_ROOT / "artifacts" / "data" / "test.csv"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"

TARGET_COLUMN = "ProdTaken"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Training data not prepared. Run scripts/prepare_data.py first.")

    train_df, test_df = load_data()
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = [col for col in X_train.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / "best_model.joblib")

    metadata = {
        "model_name": "Random Forest Classifier",
        "target_column": TARGET_COLUMN,
        "feature_columns": list(X_train.columns),
        "classification_threshold": 0.50,
        "test_accuracy": float(accuracy),
    }

    with (MODELS_DIR / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with (REPORTS_DIR / "training_report.json").open("w", encoding="utf-8") as file:
        json.dump({"test_accuracy": float(accuracy), "status": "passed"}, file, indent=2)

    print(f"Training complete. Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
