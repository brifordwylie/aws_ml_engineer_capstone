# Model: HistGradientBoosting with MAPIE Conformalized Quantile Regression
from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Template Placeholders
TEMPLATE_PARAMS = {
    "features": ['molwt', 'mollogp', 'molmr', 'heavyatomcount', 'numhacceptors', 'numhdonors', 'numheteroatoms', 'numrotatablebonds', 'numvalenceelectrons', 'numaromaticrings', 'numsaturatedrings', 'numaliphaticrings', 'ringcount', 'tpsa', 'labuteasa', 'balabanj', 'bertzct'],
    "target": "solubility",
    "train_all_data": False
}

from io import StringIO
import json
import argparse
import joblib
import os
import pandas as pd


# Function to check if dataframe is empty
def check_dataframe(df: pd.DataFrame, df_name: str) -> None:
    """Check if the DataFrame is empty and raise an error if so."""
    if df.empty:
        msg = f"*** The training data {df_name} has 0 rows! ***STOPPING***"
        print(msg)
        raise ValueError(msg)


# Function to match DataFrame columns to model features (case-insensitive)
def match_features_case_insensitive(df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """Match and rename DataFrame columns to match the model's features, case-insensitively."""
    # Create a set of exact matches from the DataFrame columns
    exact_match_set = set(df.columns)

    # Create a case-insensitive map of DataFrame columns
    column_map = {col.lower(): col for col in df.columns}
    rename_dict = {}

    # Build a dictionary for renaming columns based on case-insensitive matching
    for feature in model_features:
        if feature in exact_match_set:
            rename_dict[feature] = feature
        elif feature.lower() in column_map:
            rename_dict[column_map[feature.lower()]] = feature

    # Rename columns in the DataFrame to match model features
    return df.rename(columns=rename_dict)


# TRAINING SECTION
#
# This section (__main__) is where SageMaker will execute the training job
# and save the model artifacts to the model directory.
#
if __name__ == "__main__":
    # Template Parameters
    features = TEMPLATE_PARAMS["features"]
    target = TEMPLATE_PARAMS["target"]
    train_all_data = TEMPLATE_PARAMS["train_all_data"]
    validation_split = 0.2

    # Script arguments for input/output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    args = parser.parse_args()

    # Load training data from the specified directory
    training_files = [
        os.path.join(args.train, file)
        for file in os.listdir(args.train) if file.endswith(".csv")
    ]
    df = pd.concat([pd.read_csv(file, engine="python") for file in training_files])

    # Check if the DataFrame is empty
    check_dataframe(df, "training_df")

    # Training data split logic
    if train_all_data:
        # Use all data for both training and validation
        print("Training on all data...")
        df_train = df.copy()
        df_val = df.copy()
    elif "training" in df.columns:
        # Split data based on a 'training' column if it exists
        print("Splitting data based on 'training' column...")
        df_train = df[df["training"]].copy()
        df_val = df[~df["training"]].copy()
    else:
        # Perform a random split if no 'training' column is found
        print("Splitting data randomly...")
        df_train, df_val = train_test_split(df, test_size=validation_split, random_state=42)

    # Create HistGradientBoosting base model configured for quantile regression
    base_estimator = HistGradientBoostingRegressor(
        loss='quantile',  # Required for MAPIE CQR
        quantile=0.5,  # Will be overridden by MAPIE for different quantiles
        max_iter=1000,
        max_depth=6,
        learning_rate=0.01,
        random_state=42
    )

    # Create MAPIE CQR predictor - it will create quantile versions internally
    model = MapieQuantileRegressor(
        estimator=base_estimator,
        method="quantile",  # Only valid method for CQR
        cv="split",  # Use train/calibration split
        alpha=0.1  # For 90% coverage
    )

    # Prepare features and targets for training
    X_train = df_train[features]
    X_val = df_val[features]
    y_train = df_train[target]
    y_val = df_val[target]

    # Fit the MAPIE CQR model (train/calibration is handled internally)
    model.fit(X_train, y_train)

    # Save the trained model and any necessary assets
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    # Save the feature list to validate input during predictions
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as fp:
        json.dump(features, fp)


#
# Inference Section
#
def model_fn(model_dir):
    """Load and return the model from the specified directory."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(input_data, content_type):
    """Parse input data and return a DataFrame."""
    if not input_data:
        raise ValueError("Empty input data is not supported!")

    # Decode bytes to string if necessary
    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")

    if "text/csv" in content_type:
        return pd.read_csv(StringIO(input_data))
    elif "application/json" in content_type:
        return pd.DataFrame(json.loads(input_data))  # Assumes JSON array of records
    else:
        raise ValueError(f"{content_type} not supported!")


def output_fn(output_df, accept_type):
    """Supports both CSV and JSON output formats."""
    if "text/csv" in accept_type:
        csv_output = output_df.fillna("N/A").to_csv(index=False)  # CSV with N/A for missing values
        return csv_output, "text/csv"
    elif "application/json" in accept_type:
        return output_df.to_json(orient="records"), "application/json"  # JSON array of records (NaNs -> null)
    else:
        raise RuntimeError(f"{accept_type} accept type is not supported by this script.")


def predict_fn(df, model):
    """Make predictions using MAPIE CQR and return the DataFrame with results."""
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    # Load feature columns from the saved file
    with open(os.path.join(model_dir, "feature_columns.json")) as fp:
        model_features = json.load(fp)

    # Match features in a case-insensitive manner
    matched_df = match_features_case_insensitive(df, model_features)

    # Get CQR predictions - returns point prediction and intervals
    X_pred = matched_df[model_features]
    y_pred, y_pis = model.predict(X_pred)

    # Add predictions to dataframe
    df["prediction"] = y_pred

    # CQR gives adaptive intervals
    df["q_05"] = y_pis[:, 0, 0]  # Lower bound (5th percentile)
    df["q_95"] = y_pis[:, 1, 0]  # Upper bound (95th percentile)

    # Calculate interval width and std estimate
    interval_width = df["q_95"] - df["q_05"]
    df["prediction_std"] = interval_width / 3.29

    # For compatibility, approximate other percentiles using interval scaling
    # These are rough approximations since CQR primarily gives 5th/95th
    df["q_25"] = y_pred - interval_width * 0.25
    df["q_75"] = y_pred + interval_width * 0.25
    df["q_10"] = y_pred - interval_width * 0.4
    df["q_90"] = y_pred + interval_width * 0.4

    # IQR and IDR calculations
    df["iqr"] = df["q_75"] - df["q_25"]
    df["idr"] = df["q_90"] - df["q_10"]

    # Return the modified DataFrame
    return df