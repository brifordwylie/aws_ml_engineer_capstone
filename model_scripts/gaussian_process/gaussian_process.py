# Model: Gaussian Process Regressor with Standard Deviation output
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Kernel that should handle noisy, non-smooth data
kernel = (ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
          Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) +
          WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

# Template Placeholders
TEMPLATE_PARAMS = {
    "features": ['molwt', 'mollogp', 'molmr', 'heavyatomcount', 'numhacceptors', 'numhdonors', 'numheteroatoms', 'numrotatablebonds', 'numvalenceelectrons', 'numaromaticrings', 'numsaturatedrings', 'numaliphaticrings', 'ringcount', 'tpsa', 'labuteasa', 'balabanj', 'bertzct'],
    "target": "solubility",
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

    # Grab the features and targets for training
    X_train = df[features]
    y_train = df[target]

    # Create and train the Regression/Confidence model
    model = GaussianProcessRegressor(kernel=kernel)

    # Create a Pipeline with StandardScaler
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Save the trained model and any necessary assets
    joblib.dump(model, os.path.join(args.model_dir, "scikit_model.joblib"))

    # Save the feature list to validate input during predictions
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as fp:
        json.dump(features, fp)


#
# Inference Section
#
def model_fn(model_dir):
    """Load and return the model from the specified directory."""
    return joblib.load(os.path.join(model_dir, "scikit_model.joblib"))


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
    """Make predictions or apply transformations using the model and return the DataFrame with results."""
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    # Load feature columns from the saved file
    with open(os.path.join(model_dir, "feature_columns.json")) as fp:
        model_features = json.load(fp)

    # Match features in a case-insensitive manner
    matched_df = match_features_case_insensitive(df, model_features)

    # Run predictions and store the results
    y_mean, y_std = model.predict(matched_df[model_features], return_std=True)
    df["prediction"] = y_mean
    df["prediction_std"] = y_std

    # Return the modified DataFrame
    return df
