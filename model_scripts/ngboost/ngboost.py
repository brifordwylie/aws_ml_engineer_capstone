# Model: NGBoost Regressor with Distribution output
from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split

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

    # Create and train the NGBoost Regression model optimized for noisy molecular data
    model = NGBRegressor(
        n_estimators=1000,  # More trees for noisy data (default 500)
        learning_rate=0.01,  # Slower learning to avoid overfitting (default 0.01)
        minibatch_frac=0.8,  # Use 80% of data per iteration for robustness (default 1.0)
        col_sample=0.8,  # Use 80% of features per tree (helps with 17 features)
        tol=1e-5,  # Tighter convergence tolerance (default 1e-4)
        random_state=42,  # For reproducibility
        verbose=False
    )

    # Prepare features and targets for training
    X_train = df_train[features]
    X_val = df_val[features]
    y_train = df_train[target]
    y_val = df_val[target]

    # Train the model using the training data
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
    """Make predictions or apply transformations using the model and return the DataFrame with results."""
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    # Load feature columns from the saved file
    with open(os.path.join(model_dir, "feature_columns.json")) as fp:
        model_features = json.load(fp)

    # Match features in a case-insensitive manner
    matched_df = match_features_case_insensitive(df, model_features)

    # NGBoost predict returns distribution objects
    y_dists = model.pred_dist(matched_df[model_features])

    # Extract parameters from distribution
    dist_params = y_dists.params

    # Extract mean and std from distribution parameters
    df["prediction"] = dist_params['loc']  # mean
    df["prediction_std"] = dist_params['scale']  # standard deviation

    # Add prediction intervals using ppf (percent point function)
    df["q_05"] = y_dists.ppf(0.05)  # 5th percentile
    df["q_95"] = y_dists.ppf(0.95)  # 95th percentile

    # Add IQR (Interquartile Range): 75th - 25th percentile
    df["q_25"] = y_dists.ppf(0.25)  # 25th percentile
    df["q_75"] = y_dists.ppf(0.75)  # 75th percentile
    df["iqr"] = df["q_75"] - df["q_25"]

    # Add IDR (Interdecile Range): 90th - 10th percentile
    df["q_10"] = y_dists.ppf(0.10)  # 10th percentile
    df["q_90"] = y_dists.ppf(0.90)  # 90th percentile
    df["idr"] = df["q_90"] - df["q_10"]

    # Return the modified DataFrame
    return df
