"""Model training and evaluation pipeline for real estate price prediction.

This module handles the complete ML workflow:
- Loading and exploring data
- Feature engineering (interactions, transformations, log scaling)
- Preprocessing pipeline (imputation, scaling, encoding)
- Model training (Random Forest + Ridge Regression ensemble)
- Evaluation (metrics calculation and model persistence)

Example:
    >>> python train.py
    Trains the model and saves to models/model.pkl with metrics.json
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
import joblib
import json
from datetime import datetime
import warnings
from config import (
    DATA_FILE,
    MODEL_FILE,
    METRICS_FILE,
    CURRENT_YEAR,
    KNN_NEIGHBORS,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_RANDOM_STATE,
    TEST_SIZE,
    CROSS_VAL_FOLDS,
    OUTLIER_QUANTILES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_VARIABLE,
)

warnings.filterwarnings("ignore")


def load_data(path):
    """Load and display summary of housing dataset.

    Reads CSV file and prints basic statistics about the loaded data.

    Args:
        path (str): Path to the CSV file containing property data.

    Returns:
        pd.DataFrame: Loaded dataset with raw data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.ParserError: If the CSV file is malformed.
    """
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def engineer_features(df):
    """Create new features through interactions, transformations, and domain expertise.

    Performs feature engineering including:
    - Median imputation for missing numeric values
    - Polynomial features (interactions and squares)
    - Temporal features (age, years since remodel)
    - Ratio features (efficiency metrics)
    - Log transformations for skewed distributions

    Args:
        df (pd.DataFrame): Raw property data with original features.

    Returns:
        pd.DataFrame: Dataset with engineered features added.

    Note:
        Creates ~14 new features from domain knowledge about real estate.
        Uses CURRENT_YEAR from config for age calculations.

    Example:
        >>> df_raw = load_data('data/ames.csv')
        >>> df_featured = engineer_features(df_raw)
        >>> print(df_featured.shape)  # More columns than df_raw
    """
    df = df.copy()

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Basic interactions and features
    if "Gr Liv Area" in df.columns and "Overall Qual" in df.columns:
        df["Quality_Area"] = df["Gr Liv Area"] * df["Overall Qual"]
        df["Quality_Area_Squared"] = df["Quality_Area"] ** 2

    if "Year Built" in df.columns:
        df["House_Age"] = CURRENT_YEAR - df["Year Built"]
        df["House_Age_Squared"] = df["House_Age"] ** 2

    if "Year Remod/Add" in df.columns:
        df["Years_Since_Remodel"] = CURRENT_YEAR - df["Year Remod/Add"]

    if "Total Bsmt SF" in df.columns and "Gr Liv Area" in df.columns:
        df["Has_Basement"] = (df["Total Bsmt SF"] > 0).astype(int)
        df["Basement_Ratio"] = df["Total Bsmt SF"] / (df["Gr Liv Area"] + 1)

    if "Garage Cars" in df.columns and "Garage Area" in df.columns:
        df["Has_Garage"] = (df["Garage Cars"] > 0).astype(int)
        df["Garage_Efficiency"] = df["Garage Area"] / (df["Garage Cars"] + 1)

    if "Overall Qual" in df.columns and "Overall Cond" in df.columns:
        df["Quality_Condition_Score"] = df["Overall Qual"] * df["Overall Cond"]

    if "1st Flr SF" in df.columns and "2nd Flr SF" in df.columns:
        df["Total_Floor_Area"] = df["1st Flr SF"] + df["2nd Flr SF"]

    # Additional polynomial features
    if "Gr Liv Area" in df.columns:
        df["Gr_Liv_Area_Squared"] = df["Gr Liv Area"] ** 2
        df["Gr_Liv_Area_Cubed"] = df["Gr Liv Area"] ** 3

    if "Overall Qual" in df.columns:
        df["Overall_Qual_Squared"] = df["Overall Qual"] ** 2

    # Log transforms for skewed features
    for col in ["Gr Liv Area", "Total Bsmt SF", "Lot Area"]:
        if col in df.columns:
            df[f"{col}_Log"] = np.log1p(df[col])

    return df


def prepare_features(df):
    """Select, validate, and prepare features for model training.

    Filters to relevant numeric and categorical features, removes outliers,
    and separates features from target variable.

    Args:
        df (pd.DataFrame): Dataset with engineered features.

    Returns:
        tuple: (X, y, numeric_features, categorical_features)
            - X (pd.DataFrame): Feature matrix (numeric + categorical)
            - y (pd.Series): Target variable (SalePrice)
            - numeric_features (list): Names of numeric feature columns
            - categorical_features (list): Names of categorical feature columns

    Notes:
        - Removes rows with missing target variable
        - Filters features to those specified in config.NUMERIC_FEATURES
        - Removes outliers in price (top/bottom 1%) for robust training
        - Only keeps features that exist in the dataset

    Example:
        >>> df = engineer_features(load_data('data/ames.csv'))
        >>> X, y, num_feat, cat_feat = prepare_features(df)
        >>> print(f"Training with {X.shape[0]} samples, {X.shape[1]} features")
    """
    df = df.dropna(subset=[TARGET_VARIABLE])

    numeric_features = NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES

    numeric_features = [
        f
        for f in numeric_features
        if f in df.columns and df[f].dtype in ["int64", "float64"]
    ]
    categorical_features = [f for f in categorical_features if f in df.columns]

    X = df[numeric_features + categorical_features]
    y = df[TARGET_VARIABLE]

    # Stricter outlier removal - top and bottom quantiles (config: 1%)
    Q1, Q3 = y.quantile(OUTLIER_QUANTILES[0]), y.quantile(OUTLIER_QUANTILES[1])
    mask = (y >= Q1) & (y <= Q3)
    X, y = X[mask], y[mask]

    print(
        f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical"
    )
    print(f"Samples: {len(X)} | Price range: ${y.min():,.0f} - ${y.max():,.0f}")

    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features):
    """Construct preprocessing and modeling pipeline.

    Creates a scikit-learn Pipeline that:
    1. Preprocesses numeric features (KNN imputation + scaling)
    2. Preprocesses categorical features (one-hot encoding)
    3. Trains an ensemble model (Random Forest + Ridge Regression)

    Args:
        numeric_features (list): Column names for numeric features
        categorical_features (list): Column names for categorical features

    Returns:
        sklearn.pipeline.Pipeline: Complete preprocessing + model pipeline

    Pipeline Architecture:
        - Numeric: KNNImputer (k={}) -> StandardScaler
        - Categorical: OneHotEncoder (min_frequency=2)
        - Model: VotingRegressor (90% RF, 10% Ridge)

    Configuration:
        Uses hyperparameters from config:
        - RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE
        - Ridge alpha from config

    Example:
        >>> pipeline = build_pipeline(numeric_features, categorical_features)
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """.format(KNN_NEIGHBORS)
    numeric_pipeline = Pipeline(
        [
            ("imputer", KNNImputer(n_neighbors=KNN_NEIGHBORS)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, min_frequency=2
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # Random Forest from config
    rf_model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
        bootstrap=True,
        warm_start=False,
    )

    # Ridge with optimized alpha
    ridge_model = Ridge(alpha=0.1)

    # Weighted ensemble
    voting_model = VotingRegressor(
        [("rf", rf_model), ("ridge", ridge_model)], weights=[0.9, 0.1]
    )

    model = Pipeline([("preprocessor", preprocessor), ("regressor", voting_model)])

    return model


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """Train model and evaluate on train/test sets with cross-validation.

    Trains the pipeline model and computes comprehensive evaluation metrics
    including MAE, RMSE, R², MAPE, and cross-validation scores.

    Args:
        model (sklearn.pipeline.Pipeline): Complete preprocessing + model pipeline
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target values
        y_test (pd.Series): Test target values

    Returns:
        tuple: (trained_model, metrics_dict)
            - trained_model: Fitted pipeline model
            - metrics_dict: Dict with MAE, RMSE, R², MAPE, accuracy, precision, CV scores

    Evaluation Metrics:
        - MAE: Mean Absolute Error (in dollars)
        - RMSE: Root Mean Squared Error (in dollars)
        - R² Score: Coefficient of determination
        - MAPE: Mean Absolute Percentage Error
        - Accuracy: 100 * (1 - MAPE)
        - Precision: R² * 100
        - Cross-validation: 5-fold R² scores

    Note:
        Cross-validation uses CROSS_VAL_FOLDS from config.
    """
    print("\nTraining ensemble model...")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=CROSS_VAL_FOLDS, scoring="r2"
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<15} {'Train':<20} {'Test':<20}")
    print("-" * 60)
    print(f"{'MAE':<15} ${train_mae:>18,.0f} ${test_mae:>18,.0f}")
    print(f"{'RMSE':<15} ${train_rmse:>18,.0f} ${test_rmse:>18,.0f}")
    print(f"{'R² Score':<15} {train_r2:>19.4f} {test_r2:>19.4f}")
    print("-" * 60)
    print(
        f"\nCross-Validation (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
    )
    for i, s in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {s:.4f}")

    accuracy = 100 * (1 - test_mape)
    precision = test_r2 * 100

    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print("=" * 60)

    return model, {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_mape": test_mape,
        "accuracy": accuracy,
        "precision": precision,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


def save_model(model, metrics, X_train, y_train):
    """Save trained model and metrics to disk.

    Persists the trained model and comprehensive metadata/metrics to files
    for later loading and inference.

    Args:
        model (sklearn.pipeline.Pipeline): Trained model pipeline
        metrics (dict): Evaluation metrics from train_and_evaluate()
        X_train (pd.DataFrame): Training features (for shape info)
        y_train (pd.Series): Training targets (for size info)

    Output Files:
        - models/model.pkl: Serialized trained model (via joblib)
        - models/metrics.json: JSON file with all evaluation metrics
        - models/model_info.txt: Human-readable model summary

    Metadata Saved:
        - Train size, number of features, timestamp
        - Model name and description
        - Test metrics (R², MAE, RMSE, MAPE, accuracy, precision)
        - Cross-validation scores

    Example:
        >>> trained_model, metrics = train_and_evaluate(...)
        >>> save_model(trained_model, metrics, X_train, y_train)
        ✓ Model and metrics saved to models/
    """
    joblib.dump(model, str(MODEL_FILE))

    metrics["train_size"] = len(X_train)
    metrics["num_features"] = X_train.shape[1]
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["model_name"] = "Random Forest + Ridge Regression Ensemble"
    metrics["description"] = "Weighted voting ensemble (90% RF, 10% Ridge)"

    with open(str(METRICS_FILE), "w") as f:
        json.dump(metrics, f, indent=2)

    with open("models/model_info.txt", "w") as f:
        f.write(f"Model: Random Forest + Ridge Regression Ensemble\n")
        f.write(f"RF Estimators: {RF_N_ESTIMATORS} | Max Depth: {RF_MAX_DEPTH}\n")
        f.write(f"Weights: 90% RF, 10% Ridge\n")
        f.write(f"Test R²: {metrics['test_r2']:.4f}\n")
        f.write(f"Test Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Test Precision: {metrics['precision']:.2f}%\n")

    print("\nModel saved!")


def main():
    """Execute complete model training pipeline.

    Orchestrates the full workflow:
    1. Load data from CSV
    2. Engineer features (interactions, transformations)
    3. Prepare features (select, validate, remove outliers)
    4. Split into train/test sets
    5. Build preprocessing + model pipeline
    6. Train and evaluate
    7. Save model and metrics

    Uses configuration from config.py for paths and hyperparameters.

    Example:
        $ python train.py
        Loads data/ames.csv, trains model, saves to models/
    """
    print("=" * 60)
    print("REAL ESTATE PRICE PREDICTION")
    print("=" * 60)

    df = load_data(str(DATA_FILE))
    df = engineer_features(df)
    X, y, num_feat, cat_feat = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RF_RANDOM_STATE
    )

    print(f"Train/Test Split: {len(X_train)} / {len(X_test)}")

    model = build_pipeline(num_feat, cat_feat)
    trained_model, metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test)

    save_model(trained_model, metrics, X_train, y_train)


if __name__ == "__main__":
    main()
