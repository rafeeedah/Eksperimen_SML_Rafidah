import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# =========================
# CONFIG
# =========================
RAW_DATA_PATH = "german_credit_data.csv"
OUTPUT_DIR = "preprocessing/german_credit_data_preprocessing"
RANDOM_STATE = 88
TEST_SIZE = 0.2

TARGET_COL = "credit_risk"

CATEGORICAL_COLS = [
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

NUMERICAL_COLS = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "num_dependents",
]

# =========================
# LOAD DATA
# =========================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# =========================
# PREPROCESSING
# =========================
def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )

    return preprocessor


# =========================
# MAIN PIPELINE
# =========================
def preprocess_and_split(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )


# =========================
# SAVE OUTPUTS
# =========================
def save_outputs(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(X_train, f"{output_dir}/X_train.joblib")
    joblib.dump(X_test, f"{output_dir}/X_test.joblib")
    joblib.dump(y_train, f"{output_dir}/y_train.joblib")
    joblib.dump(y_test, f"{output_dir}/y_test.joblib")
    joblib.dump(preprocessor, f"{output_dir}/preprocessor.joblib")


# =========================
# ENTRY POINT
# =========================
def main():
    df = load_data(RAW_DATA_PATH)

    (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    ) = preprocess_and_split(df)

    save_outputs(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
