# Template Placeholders
TEMPLATE_PARAMS = {
    "model_type": "quantile_regressor",
    "target_column": "solubility",
    "feature_list": ['molwt', 'mollogp', 'molmr', 'heavyatomcount', 'numhacceptors', 'numhdonors', 'numheteroatoms', 'numrotatablebonds', 'numvalenceelectrons', 'numaromaticrings', 'numsaturatedrings', 'numaliphaticrings', 'ringcount', 'tpsa', 'labuteasa', 'balabanj', 'bertzct'],
    "model_metrics_s3_path": "s3://sandbox-sageworks-artifacts/models/training/aqsol-quantiles",
    "train_all_data": False
}

# Imports for XGB Model
import xgboost as xgb
import awswrangler as wr

# Model Performance Scores
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error
)

from io import StringIO
import json
import argparse
import os
import pandas as pd


# Function to check if dataframe is empty
def check_dataframe(df: pd.DataFrame, df_name: str) -> None:
    """
    Check if the provided dataframe is empty and raise an exception if it is.

    Args:
        df (pd.DataFrame): DataFrame to check
        df_name (str): Name of the DataFrame
    """
    if df.empty:
        msg = f"*** The training data {df_name} has 0 rows! ***STOPPING***"
        print(msg)
        raise ValueError(msg)

def match_features_case_insensitive(df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """
    Matches and renames the DataFrame's column names to match the model's feature names (case-insensitive).
    Prioritizes exact case matches first, then falls back to case-insensitive matching if no exact match exists.

    Args:
        df (pd.DataFrame): The DataFrame with the original columns.
        model_features (list): The desired list of feature names (mixed case allowed).

    Returns:
        pd.DataFrame: The DataFrame with renamed columns to match the model's feature names.
    """
    # Create a mapping for exact and case-insensitive matching
    exact_match_set = set(df.columns)
    column_map = {}

    # Build the case-insensitive map (if we have any duplicate columns, the first one wins)
    for col in df.columns:
        lower_col = col.lower()
        if lower_col not in column_map:
            column_map[lower_col] = col

    # Create a dictionary for renaming
    rename_dict = {}
    for feature in model_features:
        # Check for an exact match first
        if feature in exact_match_set:
            rename_dict[feature] = feature

        # If not an exact match, fall back to case-insensitive matching
        elif feature.lower() in column_map:
            rename_dict[column_map[feature.lower()]] = feature

    # Rename the columns in the DataFrame to match the model's feature names
    return df.rename(columns=rename_dict)


if __name__ == "__main__":
    """The main function is for training the XGBoost Quantile Regression models"""

    # Harness Template Parameters
    target = TEMPLATE_PARAMS["target_column"]
    feature_list = TEMPLATE_PARAMS["feature_list"]
    model_metrics_s3_path = TEMPLATE_PARAMS["model_metrics_s3_path"]
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    q_models = {}

    # Script arguments for input/output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    args = parser.parse_args()

    # Read the training data into DataFrames
    training_files = [
        os.path.join(args.train, file)
        for file in os.listdir(args.train)
        if file.endswith(".csv")
    ]
    print(f"Training Files: {training_files}")

    # Combine files and read them all into a single pandas dataframe
    df = pd.concat([pd.read_csv(file, engine="python") for file in training_files])

    # Check if the dataframe is empty
    check_dataframe(df, "training_df")

    # Features/Target output
    print(f"Target: {target}")
    print(f"Features: {str(feature_list)}")
    print(f"Data Shape: {df.shape}")

    # Grab our Features and Target with traditional X, y handles
    y = df[target]
    X = df[feature_list]

    # Train models for each of the quantiles
    for q in quantiles:
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": q,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)

        # Convert quantile to string
        q_str = f"q_{int(q * 100):02}"

        # Store the model
        q_models[q_str] = model

    # Train a model for RMSE predictions
    params = {"objective": "reg:squarederror"}
    rmse_model = xgb.XGBRegressor(**params)
    rmse_model.fit(X, y)

    # Run predictions for each quantile
    quantile_predictions = {q: model.predict(X) for q, model in q_models.items()}

    # Create a copy of the provided DataFrame and add the new columns
    result_df = df[[target]].copy()

    # Add the quantile predictions to the DataFrame
    for name, preds in quantile_predictions.items():
        result_df[name] = preds

    # Add the Inner Quartile Range (IQR) to the DataFrame
    result_df["iqr"] = result_df["q_75"] - result_df["q_25"]

    # Add the Inter Decile Range (IDR) range between the 10th and 90th percentiles
    result_df["idr"] = result_df["q_90"] - result_df["q_10"]

    # Add the RMSE predictions to the DataFrame
    result_df["prediction"] = rmse_model.predict(X)

    # Now compute residuals on the rmse prediction
    result_df["residual"] = result_df[target] - result_df["prediction"]
    result_df["residual_abs"] = result_df["residual"].abs()


    # Save the results dataframe to S3
    wr.s3.to_csv(
        result_df,
        path=f"{model_metrics_s3_path}/validation_predictions.csv",
        index=False,
    )

    # Report Performance Metrics
    rmse = root_mean_squared_error(result_df[target], result_df["prediction"])
    mae = mean_absolute_error(result_df[target], result_df["prediction"])
    r2 = r2_score(result_df[target], result_df["prediction"])
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"NumRows: {len(result_df)}")

    # Now save the quantile models
    for name, model in q_models.items():
        model_path = os.path.join(args.model_dir, f"{name}.json")
        print(f"Saving model:  {model_path}")
        model.save_model(model_path)

    # Save the RMSE model
    model_path = os.path.join(args.model_dir, "rmse.json")
    print(f"Saving model:  {model_path}")
    rmse_model.save_model(model_path)

    # Also save the features (this will validate input during predictions)
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as fp:
        json.dump(feature_list, fp)


def model_fn(model_dir) -> dict:
    """Deserialized and return all the fitted models from the model directory.

    Args:
        model_dir (str): The directory where the models are stored.

    Returns:
        dict: A dictionary of the models.
    """

    # Load ALL the Quantile models from the model directory
    models = {}
    for file in os.listdir(model_dir):
        if file.startswith("q") and file.endswith(".json"):  # The Quantile models
            # Load the model
            model_path = os.path.join(model_dir, file)
            print(f"Loading model: {model_path}")
            model = xgb.XGBRegressor()
            model.load_model(model_path)

            # Store the quantile model
            q_name = os.path.splitext(file)[0]
            models[q_name] = model

    # Now load the RMSE model
    model_path = os.path.join(model_dir, "rmse.json")
    print(f"Loading model: {model_path}")
    models["rsme"] = xgb.XGBRegressor()
    models["rsme"].load_model(model_path)

    # Return all the models
    return models


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


def predict_fn(df, models) -> pd.DataFrame:
    """Make Predictions with our XGB Quantile Regression Model

    Args:
        df (pd.DataFrame): The input DataFrame
        models (dict): The dictionary of models to use for predictions

    Returns:
        pd.DataFrame: The DataFrame with the predictions added
    """

    # Grab our feature columns (from training)
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    with open(os.path.join(model_dir, "feature_columns.json")) as fp:
        model_features = json.load(fp)
    print(f"Model Features: {model_features}")

    # We're going match features in a case-insensitive manner, accounting for all the permutations
    # - Model has a feature list that's any case ("Id", "taCos", "cOunT", "likes_tacos")
    # - Incoming data has columns that are mixed case ("ID", "Tacos", "Count", "Likes_Tacos")
    matched_df = match_features_case_insensitive(df, model_features)

    # Predict the features against all the models
    for name, model in models.items():
        if name == "rsme":
            df["prediction"] = model.predict(matched_df[model_features])
        else:
            df[name] = model.predict(matched_df[model_features])

    # Add the Inner Quartile Range (IQR) to the DataFrame
    df["iqr"] = df["q_75"] - df["q_25"]

    # Add the Inter Decile Range (IDR) range between the 10th and 90th percentiles
    df["idr"] = df["q_90"] - df["q_10"]

    # Reorganize the columns so they are in alphabetical order
    df = df.reindex(sorted(df.columns), axis=1)

    # All done, return the DataFrame
    return df
