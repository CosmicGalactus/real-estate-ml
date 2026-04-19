"""Configuration and constants for Real Estate ML System

This module centralizes all configuration values, hardcoded constants, and settings
used across the application. This follows the principle of avoiding hardcoding by
storing all magic numbers, paths, and configuration in one place.

Configuration Categories:
    - Paths: File and directory locations
    - Model: ML model hyperparameters
    - UI: UI/UX constants and defaults
    - Data: Data processing constants
    - Feature: Feature engineering parameters
"""

from pathlib import Path
from typing import List

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# Data file paths
DATA_FILE = DATA_DIR / "ames.csv"
MODEL_FILE = MODELS_DIR / "model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Random Forest hyperparameters
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 22
RF_RANDOM_STATE = 42
RF_N_JOBS = -1

# Linear Regression for ensemble
LR_RANDOM_STATE = 42

# Ridge Regression for ensemble
RIDGE_ALPHA = 1.0

# Model training/testing
TEST_SIZE = 0.2
CROSS_VAL_FOLDS = 5
OUTLIER_QUANTILES = (0.01, 0.99)  # Remove top and bottom 1%

# KNN Imputation
KNN_NEIGHBORS = 5

# ============================================================================
# UI CONFIGURATION
# ============================================================================

# Page config
APP_TITLE = "Real Estate Price Predictor"
APP_ICON = "🏡"
APP_LAYOUT = "wide"

# Neighborhoods (drop-down options)
NEIGHBORHOODS = ["Northridge", "Westside", "Downtown", "Suburbs"]

# Property characteristics ranges
SQFT_MIN = 500
SQFT_MAX = 6000
SQFT_DEFAULT = 2000
SQFT_STEP = 100

BEDROOMS_MIN = 1
BEDROOMS_MAX = 6
BEDROOMS_DEFAULT = 3

BATHROOMS_MIN = 1
BATHROOMS_MAX = 5
BATHROOMS_DEFAULT = 2

GARAGE_CARS_MIN = 0
GARAGE_CARS_MAX = 4
GARAGE_CARS_DEFAULT = 2

QUALITY_MIN = 1
QUALITY_MAX = 10
QUALITY_DEFAULT = 7

CONDITION_MIN = 1
CONDITION_MAX = 10
CONDITION_DEFAULT = 7

YEAR_BUILT_MIN = 1800
YEAR_BUILT_MAX = 2026
YEAR_BUILT_DEFAULT = 2005

# Investment types
INVESTMENT_TYPES = ["Buy to Live", "Rental", "Flip"]
INVESTMENT_DEFAULT = "Buy to Live"

# Risk tolerance levels
RISK_LEVELS = ["Low", "Medium", "High"]
RISK_DEFAULT = "Medium"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Current year for age calculations
CURRENT_YEAR = 2026

# Property feature estimation multipliers
BASEMENT_RATIO_MULTIPLIER = 0.5
FLOOR_UPPER_RATIO = 0.3
GARAGE_SQFT_PER_CAR = 250
LOT_AREA_DEFAULT = 10000

# Categorical feature defaults
BLDG_TYPE_DEFAULT = "1Fam"
HOUSE_STYLE_DEFAULT = "2Story"

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Numeric features for model input
NUMERIC_FEATURES = [
    "Gr Liv Area",
    "Total Bsmt SF",
    "1st Flr SF",
    "Garage Area",
    "Lot Area",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "House_Age",
    "Bedroom AbvGr",
    "Full Bath",
    "Half Bath",
    "Kitchen AbvGr",
    "TotRms AbvGrd",
    "Garage Cars",
    "Quality_Area",
    "Quality_Condition_Score",
    "Total_Floor_Area",
]

# Categorical features for model input
CATEGORICAL_FEATURES = ["Neighborhood", "Bldg Type", "House Style"]

# Target variable
TARGET_VARIABLE = "SalePrice"

# ============================================================================
# RAG/KNOWLEDGE BASE CONFIGURATION
# ============================================================================

# Chroma DB collection settings
CHROMA_COLLECTION_NAME = "market_data"
CHROMA_SIMILARITY_METRIC = "cosine"
CHROMA_TOP_K_SEARCH = 3

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

# Analysis state thresholds
PRICE_VALIDATION_THRESHOLD = 0.15  # 15% deviation tolerance
RECOMMENDATION_CONFIDENCE_THRESHOLD = 0.7  # 70% confidence for recommendations

# ============================================================================
# LOGGING & DEBUGGING
# ============================================================================

LOG_LEVEL = "INFO"
DEBUG_MODE = False
