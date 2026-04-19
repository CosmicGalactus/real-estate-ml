"""
Model utilities for loading and using the trained ML model
Bridges between saved models and the advisory agent
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import json
import os
from pathlib import Path


class ModelPredictor:
    """Wrapper for the trained ML model"""

    def __init__(self, model_path: str = None):
        """
        Initialize model predictor

        Args:
            model_path: Path to saved model file (.joblib)
        """
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.model_metadata = {}

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load trained model from disk"""
        try:
            # Load the model (expects both model and pipeline to be saved)
            loaded_data = joblib.load(model_path)

            if isinstance(loaded_data, dict):
                # Assume dict with 'model' and 'pipeline' keys
                self.model = loaded_data.get("model")
                self.pipeline = loaded_data.get("pipeline")
                self.feature_names = loaded_data.get("feature_names", [])
                self.model_metadata = loaded_data.get("metadata", {})
            else:
                # Direct model
                self.model = loaded_data

            print(f"Model loaded successfully from {model_path}")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make price prediction for a property

        Args:
            features: Dict with property features

        Returns:
            Dict with prediction and confidence
        """

        if self.model is None:
            return {
                "success": False,
                "error": "Model not loaded",
                "predicted_price": None,
            }

        try:
            # Convert features dict to DataFrame row
            feature_df = pd.DataFrame([features])

            # Make prediction
            if self.pipeline:
                prediction = self.pipeline.predict(feature_df)[0]
            else:
                prediction = self.model.predict(feature_df)[0]

            # Get confidence if available
            confidence = None
            try:
                if hasattr(self.model, "predict_proba"):
                    confidence = float(np.max(self.model.predict_proba(feature_df)))
                elif hasattr(self.model, "score"):
                    confidence = (
                        float(self.model.score(feature_df, None)) if None else None
                    )
            except:
                confidence = None

            return {
                "success": True,
                "predicted_price": float(prediction),
                "confidence": confidence,
                "model_type": self.model_metadata.get("model_type", "unknown"),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "predicted_price": None}

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances if available"""

        if not hasattr(self.model, "feature_importances_"):
            return {}

        try:
            importances = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importances))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importances)}
        except:
            return {}


def prepare_property_features(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert user input to model-ready features

    Args:
        raw_input: User input with property details

    Returns:
        Dict with engineered features for model
    """

    features = {}

    # Core numeric features
    numeric_mapping = {
        "Gr Liv Area": "sqft",
        "Total Bsmt SF": "basement_sqft",
        "1st Flr SF": "first_floor_sqft",
        "Garage Area": "garage_area",
        "Lot Area": "lot_area",
        "Overall Qual": "quality",
        "Overall Cond": "condition",
        "Year Built": "year_built",
        "Year Remod/Add": "year_remodeled",
        "Bedroom AbvGr": "bedrooms",
        "Full Bath": "full_baths",
        "Half Bath": "half_baths",
        "Kitchen AbvGr": "kitchens",
        "TotRms AbvGrd": "total_rooms",
        "Garage Cars": "garage_cars",
    }

    # Map user input to feature names
    for model_feat, user_key in numeric_mapping.items():
        if user_key in raw_input:
            features[model_feat] = float(raw_input[user_key])

    # Categorical features
    categorical_mapping = {
        "Neighborhood": "neighborhood",
        "Bldg Type": "building_type",
        "House Style": "house_style",
    }

    for model_feat, user_key in categorical_mapping.items():
        if user_key in raw_input:
            features[model_feat] = raw_input[user_key]

    # Engineer additional features if core features present
    if "Year Built" in features:
        current_year = 2026
        features["House_Age"] = current_year - features["Year Built"]
        features["House_Age_Squared"] = features["House_Age"] ** 2

    if "Year Remod/Add" in features and "Year Built" in features:
        features["Years_Since_Remodel"] = (
            features["Year Remod/Add"] - features["Year Built"]
        )

    if "Overall Qual" in features and "Gr Liv Area" in features:
        features["Quality_Area"] = features["Overall Qual"] * features["Gr Liv Area"]
        features["Quality_Area_Squared"] = features["Quality_Area"] ** 2

    if "Total Bsmt SF" in features and "Gr Liv Area" in features:
        total_area = features["Total Bsmt SF"] + features["Gr Liv Area"]
        if total_area > 0:
            features["Basement_Ratio"] = features["Total Bsmt SF"] / total_area

    if "Overall Qual" in features and "Overall Cond" in features:
        features["Quality_Condition_Score"] = (
            features["Overall Qual"] * features["Overall Cond"]
        )

    if "Garage Area" in features and "Garage Cars" in features:
        if features["Garage Cars"] > 0:
            features["Garage_Efficiency"] = features["Garage Area"] / (
                features["Garage Cars"] * 400
            )

    # Log transforms for skewed features
    if "Gr Liv Area" in features and features["Gr Liv Area"] > 0:
        features["Gr Liv Area_Log"] = np.log1p(features["Gr Liv Area"])

    if "Total Bsmt SF" in features and features["Total Bsmt SF"] > 0:
        features["Total Bsmt SF_Log"] = np.log1p(features["Total Bsmt SF"])

    if "Lot Area" in features and features["Lot Area"] > 0:
        features["Lot Area_Log"] = np.log1p(features["Lot Area"])

    # Binary features
    features["Has_Basement"] = 1 if features.get("Total Bsmt SF", 0) > 0 else 0
    features["Has_Garage"] = 1 if features.get("Garage Cars", 0) > 0 else 0

    return features


def load_ames_data_sample(data_path: str) -> pd.DataFrame:
    """Load Ames dataset for reference/testing"""
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def get_property_options_from_dataset(data_path: str) -> Dict[str, list]:
    """Extract unique values for categorical features from dataset"""
    try:
        df = pd.read_csv(data_path)

        options = {
            "neighborhoods": (
                sorted(df["Neighborhood"].unique().tolist())
                if "Neighborhood" in df.columns
                else []
            ),
            "building_types": (
                sorted(df["Bldg Type"].unique().tolist())
                if "Bldg Type" in df.columns
                else []
            ),
            "house_styles": (
                sorted(df["House Style"].unique().tolist())
                if "House Style" in df.columns
                else []
            ),
            "quality_scores": list(range(1, 11)),  # 1-10 scale
            "condition_scores": list(range(1, 6)),  # 1-5 scale
        }

        return options

    except Exception as e:
        print(f"Error extracting options: {str(e)}")
        return {}


if __name__ == "__main__":
    # Test feature preparation
    test_input = {
        "address": "500 Test St",
        "neighborhood": "Northridge",
        "sqft": 2000,
        "basement_sqft": 800,
        "first_floor_sqft": 1100,
        "garage_area": 480,
        "lot_area": 8000,
        "quality": 7,
        "condition": 5,
        "year_built": 2005,
        "year_remodeled": 2015,
        "bedrooms": 3,
        "full_baths": 2,
        "half_baths": 1,
        "kitchens": 1,
        "total_rooms": 8,
        "garage_cars": 2,
        "building_type": "1Fam",
        "house_style": "2Story",
    }

    print("Testing feature preparation...")
    features = prepare_property_features(test_input)
    print(f"Number of features: {len(features)}")
    print(f"First few features: {dict(list(features.items())[:5])}")
